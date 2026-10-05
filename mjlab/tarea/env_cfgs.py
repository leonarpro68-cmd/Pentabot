"""Tarea de seguimiento de velocidad en piso plano para el Pentabot.

Parte de la configuracion base de unitree_rl_mjlab y la escala a un pentapodo de 2.5 kg,
torso a 0.2 m y servos MG995 de 0.98 N m (con 3 patas en apoyo ya trabajan al 64 % del
bloqueo, ver docs/MODELO.md): marcha de onda lenta y castigo fuerte al par.
"""

import math

from src.assets.robots import PENTABOT_ACTION_SCALE, get_pentabot_robot_cfg
from src.assets.robots.pentabot.pentabot_constants import LEGS
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)

# Marcha de onda: una pata en el aire a la vez (apoyo 80 % del ciclo), en el orden
# L1, L3, L5, L2, L4 (igual que scripts/demo_marcha.py). Fase de cada pata L1..L5.
GAIT_PERIOD = 0.8  # s
GAIT_STANCE = 0.8
GAIT_OFFSET = [0.0, 0.4, 0.8, 0.2, 0.6]


def pentabot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  # ---------------------------------------------------------------- simulacion
  # 4 ms de fisica, politica a 50 Hz (decimation 5) = PWM de 50 Hz del PCA9685.
  cfg.sim.mujoco.timestep = 0.004
  cfg.decimation = 5
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.njmax = 400
  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.entities = {"robot": get_pentabot_robot_cfg()}

  # Piso plano, sin escaner de terreno.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  # ---------------------------------------------------------------- sensores de contacto
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=FOOT_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, nonfoot_ground_cfg)

  # ---------------------------------------------------------------- acciones y observaciones
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PENTABOT_ACTION_SCALE

  cfg.observations["actor"].terms["phase"].params["period"] = GAIT_PERIOD
  cfg.observations["critic"].terms["phase"].params["period"] = GAIT_PERIOD
  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = FOOT_SITES

  # ---------------------------------------------------------------- comandos (escala del robot)
  # Robot simetrico: avanza igual en x que en y.
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.35
  twist_cmd.ranges.lin_vel_x = (-0.2, 0.2)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-0.6, 0.6)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.1, 0.1), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.3, 0.3)},
    {"step": 2000 * 24, "lin_vel_x": (-0.2, 0.2), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-0.6, 0.6)},
  ]

  # ---------------------------------------------------------------- eventos y aleatorizacion
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
    "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.4, 0.4),
  }
  cfg.events["reset_base"].params["pose_range"]["x"] = (-0.3, 0.3)
  cfg.events["reset_base"].params["pose_range"]["y"] = (-0.3, 0.3)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.1)  # TPU sobre piso (MODELO.md)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
  cfg.events["base_com"].params["ranges"] = {0: (-0.015, 0.015), 1: (-0.015, 0.015), 2: (-0.01, 0.01)}
  # Juego de engranajes del MG995 (1-2 grados) y cero del servo sin calibrar.
  cfg.events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)

  # Masas: la electronica del torso es un placeholder de 0.18 kg, sin pesar.
  cfg.events["torso_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso",)), "operation": "scale", "ranges": (0.85, 1.15)},
  )
  cfg.events["leg_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=(r"L[1-5]_.*",)), "operation": "scale", "ranges": (0.9, 1.1)},
  )
  # Servo sin identificar (kp, armadura y friccion estimados).
  cfg.events["servo_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.8, 1.2), "kd_range": (1.0, 1.0), "operation": "scale"},
  )
  cfg.events["servo_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "operation": "scale", "ranges": (0.6, 1.4)},
  )
  cfg.events["servo_armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "operation": "scale", "ranges": (0.7, 1.3)},
  )

  # ---------------------------------------------------------------- recompensas
  # std estrecho para velocidades de 0.2 m/s (el defecto de 0.5 m/s premia casi todo).
  cfg.rewards["track_linear_velocity"].params["std"] = 0.1
  cfg.rewards["track_angular_velocity"].params["std"] = 0.25
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_yaw.*": 0.35, r".*hip_pitch.*": 0.4, r".*knee.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_yaw.*": 0.45, r".*hip_pitch.*": 0.5, r".*knee.*": 0.4,
  }
  cfg.rewards["pose"].params["walking_threshold"] = 0.05
  cfg.rewards["pose"].params["running_threshold"] = 0.15

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = FOOT_SITES
  cfg.rewards["foot_clearance"].params["target_height"] = 0.03
  cfg.rewards["foot_clearance"].params["command_threshold"] = 0.05
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = FOOT_SITES
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.05
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.05
  cfg.rewards["stand_still"].params["command_threshold"] = 0.05
  cfg.rewards["foot_gait"].params["period"] = GAIT_PERIOD
  cfg.rewards["foot_gait"].params["offset"] = GAIT_OFFSET
  cfg.rewards["foot_gait"].params["threshold"] = GAIT_STANCE
  cfg.rewards["foot_gait"].params["command_threshold"] = 0.05
  # MG995 de 0.98 N m: castigar el par para no vivir cerca del bloqueo.
  cfg.rewards["torques"] = RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.02)

  # ---------------------------------------------------------------- terminaciones
  # Solo los pies pueden tocar el piso (torso, coxas, femures o tibias = caida).
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 1.0},
  )
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45.0)

  # ---------------------------------------------------------------- visor
  cfg.viewer.body_name = "torso"
  cfg.viewer.distance = 1.2
  cfg.viewer.elevation = -20.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg
