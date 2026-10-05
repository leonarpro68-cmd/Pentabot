"""Pentabot: constantes del robot para mjlab.

El modelo se lee directamente de Pentabot_MuJoCo/model/pentabot.xml (generado desde
CuerpoBase.iam), asi que si se regenera desde Inventor esta configuracion lo sigue.
Sobre el spec se hacen solo los cambios que mjlab necesita:
  - se quitan los actuadores <position> del XML (mjlab crea los suyos a partir de la cfg),
  - se quitan sensores y keyframes (mjlab usa sus propios nombres de sensor),
  - se nombran las geometrias de colision (*_collision) para los sensores de contacto.
Servo: TowerPro MG995 a 6 V (ver Pentabot_MuJoCo/docs/MODELO.md).
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF y mallas.
##

PENTABOT_DIR: Path = SRC_PATH / "assets" / "robots" / "Pentabot_MuJoCo" / "model"
PENTABOT_XML: Path = PENTABOT_DIR / "pentabot.xml"
assert PENTABOT_XML.exists()

LEGS = ("L1", "L2", "L3", "L4", "L5")


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PENTABOT_DIR / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PENTABOT_XML))
  spec.assets = get_assets(spec.meshdir)

  for elem in [*spec.actuators, *spec.sensors, *spec.keys]:
    spec.delete(elem)

  # Colisiones: pies = "Lk_foot_collision"; el resto (cascos convexos) = "<cuerpo>_collision<i>".
  for body in spec.bodies:
    i = 0
    for geom in body.geoms:
      if geom.contype == 0 and geom.conaffinity == 0:
        continue  # visual
      if geom.name.endswith("_foot"):
        geom.name = f"{geom.name}_collision"
      else:
        geom.name = f"{body.name}_collision{i}"
        i += 1

  # Sensores con los nombres que espera la tarea de velocidad.
  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_lin_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="imu_quat", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  spec.add_sensor(name="root_angmom", type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
                  objtype=mujoco.mjtObj.mjOBJ_BODY, objname="torso")
  return spec


##
# Servo MG995 (6 V).
##

# Igual que la clase "mg995" del XML: kp = 11.24 N m/rad (satura a ~5 grados de error) y
# par de bloqueo 0.981 N m. La recta par-velocidad la da el damping pasivo de la
# articulacion (0.150 N m s/rad = tau_stall / omega_0), que se conserva del XML; por eso
# el actuador no agrega damping propio.
MG995 = BuiltinPositionActuatorCfg(
  target_names_expr=(r"L[1-5]_hip_yaw", r"L[1-5]_hip_pitch", r"L[1-5]_knee"),
  stiffness=11.241,
  damping=0.0,
  effort_limit=0.981,
  armature=0.005,
  frictionloss=0.03,
)

# Latencia I2C + PWM del PCA9685 a 50 Hz: 0 a 20 ms (0-5 pasos de fisica de 4 ms).
MG995_CON_RETARDO = DelayedActuatorCfg(
  base_cfg=MG995,
  delay_target="position",
  delay_min_lag=0,
  delay_max_lag=5,
)

##
# Pose inicial.
##

# q = 0 es la pose de apoyo del CAD: femur horizontal, tibia vertical, pies a 0.344 m
# del centro. Torso a 0.199 m del suelo.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.202),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Configuracion final.
##

PENTABOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(MG995_CON_RETARDO,),
  soft_joint_pos_limit_factor=0.9,
)

# Escala de la accion: objetivo = pose_home + 0.3 * accion [rad].
PENTABOT_ACTION_SCALE: float = 0.3


def get_pentabot_robot_cfg() -> EntityCfg:
  """Nueva instancia de la configuracion del Pentabot."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=PENTABOT_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pentabot_robot_cfg())
  viewer.launch(robot.spec.compile())
