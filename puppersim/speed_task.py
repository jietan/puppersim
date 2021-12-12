"""Speed tracking forward locomotion task."""
import collections
import math
from typing import List
import numpy as np
import gin

from pybullet_envs.minitaur.envs_v2.sensors import sensor
from pybullet_envs.minitaur.envs_v2.tasks import task_interface
from pybullet_envs.minitaur.envs_v2.tasks import task_utils
from pybullet_envs.minitaur.envs_v2.tasks import terminal_conditions
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils


_PENALTY_FOR_EARLY_TERMINATION = 0
_TARGET_SPEED_LOWER_BOUND = 0.0  # meters per simulation step
_TARGET_SPEED_UPPER_BOUND = 0.4  # meters per simulation step
# Tolerance for speed vs target speed in reward.
_GAUSSIAN_CAP_DEVIATION = 0.03
# The size of the queue used to calculate average speed for rewarding.
_DEQUE_SIZE = 10


@gin.configurable
class SpeedRewardTask(sensor.BoxSpaceSensor):
  """Speed tracking forward locomotion task for an agent."""

  def __init__(self,
               terminal_condition=terminal_conditions
               .default_terminal_condition_for_minitaur,
               energy_penalty_coef=0,
               min_com_height=None,
               multiply_with_dt=False,
               ):
    """Initialize the speed tracking locomotion task.

    Args:
      terminal_condition: Condition that is checked to end each episode.
      energy_penalty_coef: Coefficient for the energy penalty that will be added
        to the reward. 0 by default.
      min_com_height: Minimum height for the center of mass of the robot that
        will be used to terminate the task. This is used to obtain task specific
        gaits and set by the config or gin files based on the task and robot.
      multiply_with_dt: If True, the target velocities are given in m/s. So we
        multiply the target velocity with dt when comparing at every timestep.
      speed_stages: List containing list of timesteps and list of target_speeds
        to calculate target speed during the episode via linear interpolation
        between these data points.
    """
    self._terminal_condition = terminal_condition
    self._last_front_vectors = []
    self._last_base_positions = []
    self._target_speed = 0.0
    self._num_step = 0
    self._min_com_height = min_com_height
    self._energy_penalty_coef = energy_penalty_coef
    if energy_penalty_coef < 0:
      raise ValueError('Energy Penalty Coefficient should be >= 0')
    self._target_speed_at_reset = 0.0
    self._multiply_with_dt = multiply_with_dt
    np.random.seed(0)

    super(SpeedRewardTask, self).__init__(
        name='Speed goal sensor',
        shape=[
            1,
        ],
        lower_bound=[_TARGET_SPEED_LOWER_BOUND],
        upper_bound=[_TARGET_SPEED_UPPER_BOUND])

  def __call__(self, env):
    """Return reward.

    Args:
      env: gym environment.

    Returns:
      float, reward.
    """
    return self.reward(env)

  def reset(self, env):
    """Reset the task.

    Reset the task, called in env.reset()

    Args:
      env: gym environment.
    """
    self._target_speed = np.random.rand() * (_TARGET_SPEED_UPPER_BOUND - _TARGET_SPEED_LOWER_BOUND) + _TARGET_SPEED_LOWER_BOUND
    print("target_speed", self._target_speed)
    self._env = env
    self._target_speed_coef = 0.0
    self._num_step = 0
    self._last_front_vectors = [] # collections.deque([], maxlen=_DEQUE_SIZE)
    self._last_base_positions = [] # collections.deque([], maxlen=_DEQUE_SIZE)
    self._last_front_vectors.append(
        self._get_robot_front_direction_on_xy_plane())
    self._last_base_positions.append(self._env.robot.base_position)

  def update(self, env):
    """Update the task, called at every time step."""
    del env
    self._num_step += 1
    self._last_front_vectors.append(
        self._get_robot_front_direction_on_xy_plane())
    self._last_base_positions.append(self._env.robot.base_position)

  def reward(self, env):
    """Calculate the reward based on desired and actual speed."""
    del env
    if self._terminal_condition(self._env):
      return _PENALTY_FOR_EARLY_TERMINATION
    reward = self._forward_reward_directional()
    # Energy
    if self._energy_penalty_coef > 0:
      energy_reward = -task_utils.calculate_estimated_energy_consumption(
          self._env.robot.motor_torques, self._env.robot.motor_velocities,
          self._env.sim_time_step, self._env.num_action_repeat)
      reward += energy_reward * self._energy_penalty_coef
    return reward

  def get_observation_datatype(self):
    """Returns the data type for the numpy structured array.

    This is a required method as this task inherits sensor.Sensor.
    """
    return [('target_speed', np.float64)]

  def get_observation(self):
    """Returns the observation data based on the desired speed.

    This is a required method as this task inherits sensor.BoxSpaceSensor.
    """
    return np.asarray([self._target_speed])

  def _forward_reward_directional(self):
    """Calculates the forward reward based on target speed and robot direction.

    Forward reward is calculated based on robot's speed and target speed in
    robot's forward movement direction. We keep the robot's latest positions and
    front directions in fixed length queues. We use the average speed in the
    reward by using the robot's oldest recorded position, projected to oldest
    recorded forward direction. This function uses gaussian distribution
    around target speed. If the robot's speed is within that deviation range, it
    gets the maximum reward. The reward gradually decreases to 0 with increased
    deviation.

    Returns:
      The forward reward based on average speed and previously faced directions.
    """
    # Calculate the average speed based on the oldest recorded position and the
    # current position divided by number of steps in between.
    current_base_position = self._env.robot.base_position
    old_position = self._last_base_positions[0]
    steps = self._env.get_time_since_reset()
    average_speed = [(current_base_position[0] - old_position[0]) / steps,
                     (current_base_position[1] - old_position[1]) / steps, 0]
    # Use the oldest front vector to calculate the speed projected to the front
    # direction of the robot recorded a while ago (up to 50 steps).
    projected_speed = np.dot(average_speed, self._last_front_vectors[0])
    # print(steps, self._env.robot.base_position, old_position, projected_speed, average_speed)
    forward_reward = math.exp(-(projected_speed - self._target_speed)**2 /
                              (2 * _GAUSSIAN_CAP_DEVIATION**2))
    return forward_reward

  def _get_robot_front_direction_on_xy_plane(self):
    """Calculate the robot's direction projected to x-y plane.

    Returns:
      3 dimensional vector as a list.
    """
    current_base_orientation = self._env.robot.base_orientation_quaternion
    rot_matrix = self._env.pybullet_client.getMatrixFromQuaternion(
        current_base_orientation)
    return [0, -1, 0]
    # return [rot_matrix[0], rot_matrix[1], 0]

  def done(self, env):
    """Checks if the episode should be finished or not."""
    del env
    position = self._env.robot.base_position
    if self._min_com_height and position[2] < self._min_com_height:
      return True
    return self._terminal_condition(self._env)

  @property
  def sensors(self) -> List[sensor.BoxSpaceSensor]:
    return [self]

