"""Pentabot-Flat-SinEncoder: politica desplegable en el robot real (MG995 sin realimentacion).

El MG995 no entrega su posicion, asi que el actor solo ve lo que existe en la ESP32:
  - IMU: velocidad angular (giroscopio) y gravedad proyectada (orientacion),
  - comando del mando (vx, vy, wz),
  - fase de la marcha (reloj propio de la ESP32),
  - sus propias acciones anteriores (los objetivos que mando a los servos),
con un historial de HISTORIAL pasos para que pueda inferir el estado de las patas.
El critico (solo se usa al entrenar) ve todo: articulaciones, velocidad lineal y contactos.

Ademas se aleatoriza lo que no se conoce del servo y de la electronica:
  - kp del servo 0.6-1.4, par maximo +-20 %, retardo de 4-32 ms,
  - cero del servo descalibrado (+-3 grados) y juego de engranajes (ruido en el objetivo),
  - IMU montado algo torcido (sesgo constante por episodio) y con 0-20 ms de retardo,
  - masa del torso (bateria y electronica sin pesar) y friccion del piso.

Orden de las 130 entradas del actor (para la ESP32). El historial es POR TERMINO, no por
paso: cada termino guarda sus 5 pasos del mas viejo (t-4) al mas nuevo (t) y luego se
concatenan los terminos (CircularBuffer de mjlab):
  [  0: 15] base_ang_vel       5 x 3   (rad/s, marco del IMU)
  [ 15: 30] projected_gravity  5 x 3   (gravedad unitaria en el marco del torso)
  [ 30: 45] command            5 x 3   (vx, vy, wz)
  [ 45: 55] phase              5 x 2   (sin, cos; 0, 0 si no se pide andar)
  [ 55:130] actions            5 x 15  (accion cruda anterior, antes de x0.3)
Al arrancar (reset) el historial se llena repitiendo la primera lectura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from mjlab.actuator import DelayedActuator
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.noise.noise_cfg import NoiseModelWithAdditiveBiasCfg

from mjlab.tasks.velocity import mdp
from src.assets.robots import PENTABOT_ACTION_SCALE

from .env_cfgs import FOOT_SITES, pentabot_flat_env_cfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

HISTORIAL = 5  # pasos de 20 ms que ve el actor (100 ms)

# Umbral unico de "se esta pidiendo andar": |v_xy| + |wz|. Lo usan la fase, la marcha,
# el despeje, el deslizamiento y la postura quieta, para que no se contradigan.
UMBRAL_MARCHA = 0.05

# MG995 a 6 V: 0.16 s / 60 grados en vacio = 6.5 rad/s. Por encima de ~5 rad/s el
# servo real ya no sigue el objetivo.
VEL_MAX_SERVO = 5.0  # rad/s
ALTURA_TORSO = 0.19  # m, de pie con q = 0 se asienta en 0.196 m


##
# Accion: el servo real no llega exacto al objetivo.
##


@dataclass(kw_only=True)
class AccionServoCfg(JointPositionActionCfg):
  """Objetivo de posicion con el juego de engranajes del MG995 (ruido por paso)."""

  ruido_objetivo: float = 0.0  # rad, uniforme +-ruido_objetivo en cada paso

  def build(self, env: ManagerBasedRlEnv) -> "AccionServo":
    return AccionServo(self, env)


class AccionServo(JointPositionAction):
  cfg: AccionServoCfg

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)
    if self.cfg.ruido_objetivo > 0.0:
      ruido = (torch.rand_like(self._processed_actions) * 2.0 - 1.0) * self.cfg.ruido_objetivo
      self._processed_actions = self._processed_actions + ruido


##
# Observaciones.
##


def _pide_andar(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  return (torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > UMBRAL_MARCHA


def fase_marcha(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  """(sin, cos) de la fase global; (0, 0) si no se pide andar.

  Igual que mdp.phase pero con el mismo umbral que la recompensa de marcha: la version
  base ponia la fase a cero por debajo de 0.1 mientras la recompensa ya exigia caminar
  desde 0.05, y a 0.2 m/s de maximo eso es buena parte de los comandos.
  """
  t = (env.episode_length_buf * env.step_dt) % period / period
  fase = torch.stack((torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)), dim=1)
  return fase * _pide_andar(env, command_name).unsqueeze(1).float()


##
# Recompensas.
##


def altura_torso_l2(env: ManagerBasedRlEnv, target_height: float) -> torch.Tensor:
  """Penaliza que el torso se hunda o se levante respecto a la altura de pie."""
  asset: Entity = env.scene["robot"]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def saturacion_servo(env: ManagerBasedRlEnv, fraccion: float) -> torch.Tensor:
  """Penaliza el par por encima de `fraccion` del limite de cada servo.

  El MG995 cerca del bloqueo se calienta, pierde precision y su par real varia mucho
  entre unidades; la politica debe dejar margen. Usa el limite aleatorizado de cada env.
  """
  asset: Entity = env.scene["robot"]
  limite = env.sim.model.actuator_forcerange[:, asset.indexing.ctrl_ids, 1]  # [B, 15]
  par = torch.abs(asset.data.actuator_force)  # [B, nu]
  exceso = torch.clamp(par / limite - fraccion, min=0.0)
  return torch.sum(torch.square(exceso / (1.0 - fraccion)), dim=1)


def velocidad_servo_excesiva(env: ManagerBasedRlEnv, limite: float) -> torch.Tensor:
  """Penaliza velocidades articulares que el MG995 real no puede dar."""
  asset: Entity = env.scene["robot"]
  return torch.sum(torch.square(torch.clamp(torch.abs(asset.data.joint_vel) - limite, min=0.0)), dim=1)


def apoyos_minimos(env: ManagerBasedRlEnv, sensor_name: str, minimo: int) -> torch.Tensor:
  """1 cuando hay menos de `minimo` pies en el suelo.

  Con 3 patas apoyadas el MG995 ya trabaja al 64 % de su par: con menos se cae en el real
  aunque en simulacion lo salve un servo optimista.
  """
  sensor = env.scene[sensor_name]
  en_suelo = (sensor.data.found > 0).sum(dim=1)
  return (en_suelo < minimo).float()


##
# Eventos.
##


def limites_par(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  rango: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Escala el par maximo de cada servo (dr.effort_limits no admite DelayedActuator)."""
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  defecto = env.sim.get_default_field("actuator_forcerange")
  for act in asset.actuators:
    base = act.base_actuator if isinstance(act, DelayedActuator) else act
    ids = base.global_ctrl_ids
    escala = torch.empty((len(env_ids), len(ids)), device=env.device).uniform_(*rango)
    env.sim.model.actuator_forcerange[env_ids[:, None], ids, 0] = defecto[ids, 0] * escala
    env.sim.model.actuator_forcerange[env_ids[:, None], ids, 1] = defecto[ids, 1] * escala


##
# Configuracion.
##


def pentabot_sin_encoder_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pentabot_flat_env_cfg(play=play)

  # ---------------------------------------------------------------- accion
  cfg.actions["joint_pos"] = AccionServoCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=PENTABOT_ACTION_SCALE,
    use_default_offset=True,
    ruido_objetivo=0.0 if play else 0.02,  # ~1 grado de juego de engranajes
  )

  # ---------------------------------------------------------------- observaciones
  periodo = cfg.observations["actor"].terms["phase"].params["period"]
  fase = ObservationTermCfg(func=fase_marcha, params={"period": periodo, "command_name": "twist"})

  # IMU: ruido blanco + sesgo constante por episodio (montaje torcido ~2 grados, deriva
  # del giroscopio) y 0-1 pasos (0-20 ms) de retardo del filtro / bus I2C.
  imu_gyro = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
    noise=NoiseModelWithAdditiveBiasCfg(
      noise_cfg=Unoise(n_min=-0.15, n_max=0.15),
      bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05),
    ),
    delay_min_lag=0,
    delay_max_lag=1,
  )
  imu_gravedad = ObservationTermCfg(
    func=mdp.projected_gravity,
    noise=NoiseModelWithAdditiveBiasCfg(
      noise_cfg=Unoise(n_min=-0.03, n_max=0.03),
      bias_noise_cfg=Unoise(n_min=-0.035, n_max=0.035),
    ),
    delay_min_lag=0,
    delay_max_lag=1,
  )

  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      "base_ang_vel": imu_gyro,
      "projected_gravity": imu_gravedad,
      "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
      "phase": fase,
      "actions": ObservationTermCfg(func=mdp.last_action),
    },
    concatenate_terms=True,
    enable_corruption=not play,
    history_length=HISTORIAL,
    flatten_history_dim=True,
  )
  # Critico privilegiado: lo mismo que antes (articulaciones, velocidad lineal, contactos,
  # altura de pies) sin ruido, con la misma fase que el actor.
  cfg.observations["critic"].terms["phase"] = fase

  # ---------------------------------------------------------------- aleatorizacion
  cfg.events["servo_gains"].params["kp_range"] = (0.6, 1.4)
  cfg.events["servo_par"] = EventTermCfg(mode="startup", func=limites_par, params={"rango": (0.8, 1.2)})
  # Cero del servo descalibrado: encoder_bias se resta del objetivo en JointPositionAction,
  # asi que aqui modela el error de calibracion del pulso de q = 0 (+-3 grados).
  cfg.events["encoder_bias"].params["bias_range"] = (-0.05, 0.05)
  cfg.events["torso_mass"].params["ranges"] = (0.8, 1.4)  # bateria y electronica sin pesar
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.2)
  cfg.events["base_com"].params["ranges"] = {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.01, 0.01)}
  cfg.events["servo_friction"].params["ranges"] = (0.5, 1.5)
  # Retardo del servo: bucle de la ESP32 + I2C + PWM de 20 ms del PCA9685 (1-8 pasos de 4 ms).
  # (copias: la cfg del actuador es un objeto de modulo compartido con Pentabot-Flat)
  robot = cfg.scene.entities["robot"]
  robot.articulation = replace(
    robot.articulation,
    actuators=tuple(replace(a, delay_min_lag=1, delay_max_lag=8) for a in robot.articulation.actuators),
  )
  # Arranque desde una pose algo distinta (en el real los servos parten de donde quedaron).
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)

  # ---------------------------------------------------------------- recompensas
  # Objetivo principal: seguir el comando del mando.
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_angular_velocity"].weight = 1.0

  # Torso plano y a su altura: el IMU real estima mejor y no se arrastra la panza.
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["altura_torso"] = RewardTermCfg(
    func=altura_torso_l2, weight=-100.0, params={"target_height": ALTURA_TORSO}
  )

  # Marcha de onda: con el MG995 conviene tener siempre 4 pies (minimo 3) en el suelo.
  cfg.rewards["foot_gait"].weight = 1.0
  cfg.rewards["apoyos_minimos"] = RewardTermCfg(
    func=apoyos_minimos, weight=-1.0, params={"sensor_name": "feet_ground_contact", "minimo": 3}
  )
  cfg.rewards["foot_clearance"].weight = -3.0
  cfg.rewards["foot_clearance"].params["target_height"] = 0.035
  cfg.rewards["foot_slip"].weight = -2.0
  cfg.rewards["soft_landing"].weight = -2e-3

  # Servos: suavidad del objetivo (en lazo abierto un salto brusco no se corrige),
  # margen de par y velocidad alcanzable.
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards["action_acc_l2"] = RewardTermCfg(func=mdp.action_acc_l2, weight=-0.05)
  cfg.rewards["torques"].weight = -0.02
  cfg.rewards["saturacion_servo"] = RewardTermCfg(
    func=saturacion_servo, weight=-0.5, params={"fraccion": 0.8}
  )
  cfg.rewards["velocidad_servo"] = RewardTermCfg(
    func=velocidad_servo_excesiva, weight=-0.1, params={"limite": VEL_MAX_SERVO}
  )

  # Mismo umbral de "andar" en todas las recompensas que dependen del comando.
  for nombre in ("foot_gait", "foot_clearance", "foot_slip", "soft_landing", "stand_still"):
    cfg.rewards[nombre].params["command_threshold"] = UMBRAL_MARCHA
  assert cfg.rewards["foot_clearance"].params["asset_cfg"].site_names == FOOT_SITES

  return cfg
