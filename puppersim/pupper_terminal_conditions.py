"""Contains the terminal conditions for locomotion tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import numpy as np

from pybullet_envs.minitaur.envs_v2.utilities import minitaur_pose_utils
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils
from pybullet_envs.minitaur.envs_v2.utilities import termination_reason as tr


@gin.configurable
def default_terminal_condition_for_pupper(env):
  """A default terminal condition for Pupper.

  Pupper is considered as fallen if the base position is too low or the base
  tilts/rolls too much.

  Args:
    env: An instance of MinitaurGymEnv

  Returns:
    A boolean indicating if Minitaur is fallen.
  """
  roll, pitch, _ = env.robot.base_roll_pitch_yaw
  pos = env_utils.get_robot_base_position(env.robot)
  return abs(roll) > 0.4 or abs(pitch) > 0.4 or pos[2] < 0.05 or pos[2] > 0.6
