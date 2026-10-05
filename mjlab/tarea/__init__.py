from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pentabot_flat_env_cfg
from .rl_cfg import pentabot_ppo_runner_cfg

register_mjlab_task(
  task_id="Pentabot-Flat",
  env_cfg=pentabot_flat_env_cfg(),
  play_env_cfg=pentabot_flat_env_cfg(play=True),
  rl_cfg=pentabot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

from .rl_cfg import pentabot_sin_encoder_ppo_runner_cfg
from .sin_encoder import pentabot_sin_encoder_env_cfg

register_mjlab_task(
  task_id="Pentabot-Flat-SinEncoder",
  env_cfg=pentabot_sin_encoder_env_cfg(),
  play_env_cfg=pentabot_sin_encoder_env_cfg(play=True),
  rl_cfg=pentabot_sin_encoder_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
