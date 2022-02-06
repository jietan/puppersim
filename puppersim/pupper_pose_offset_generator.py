"""Simple openloop trajectory generators."""

import attr
import gin
from gym import spaces
import numpy as np



@gin.configurable
class PupperPoseOffsetGenerator(object):
  """A trajectory generator that return constant motor angles."""

  def __init__(
      self,
      init_abduction=0.0,
      init_hip=0.6,
      init_knee=-1.2,
      action_limit=0.5,
      ):
    """Initializes the controller."""
    self._pose = np.array([init_abduction, init_hip, init_knee] * 4)
    action_high = np.array([action_limit] * 12)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    """Computes the trajectory according to input time and action.

    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input leg pose from a NN controller.

    Returns:
      A numpy array. The desired motor angles.
    """
    del current_time
    return self._pose + input_action

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation
