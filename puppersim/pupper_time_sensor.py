# Lint as: python3
"""The on robot sensor classes."""

from typing import Any, Callable, Iterable, Optional, Sequence, Type, Text, Tuple, Union

import gin
import gym
import numpy as np

from pybullet_envs.minitaur.envs_v2.sensors import sensor
from pybullet_envs.minitaur.envs_v2.utilities import noise_generators

@gin.configurable
class PeriodicSignalSensor(sensor.Sensor):
  """A sensor that reads motor angles from the robot."""

  def __init__(self,
               name: Text = "PeriodicSignal",
               dtype: Type[Any] = np.float64,
               noise_generator: Union[Callable[..., Any],
                                      noise_generators.NoiseGenerator] = None,
               sensor_latency: Union[float, Sequence[float]] = 0.0,
               frequencies: Sequence[float] = [1.0]):
    """Initializes the class.
    Args:
      name: The name of the sensor.
      dtype: The datatype of this sensor.
      noise_generator: Adds noise to the sensor readings.
      sensor_latency: There are two ways to use this expected sensor latency.
        For both methods, the latency should be in the same unit as the sensor
        data timestamp. 1. As a single float number, the observation will be a
        1D array. For real robots, this should be set to 0.0. 2. As a array of
        floats, the observation will be a 2D array based on how long the history
        need to be. Thus, [0.0, 0.1, 0.2] is a history length of 3.
      frequencies: Sequence of frequencies to use for generating periodic signals. Unit is [Hz]
      
    """
    super().__init__(
        name=name,
        sensor_latency=sensor_latency,
        interpolator_fn=sensor.linear_obs_blender)
    self._noise_generator = noise_generator
    self._dtype = dtype
    self._frequencies = np.array(frequencies)

  def set_robot(self, robot):
    self._robot = robot
    # Creates the observation space
    max_val = np.ones(2 * len(self._frequencies))
    self._observation_space = self._stack_space(
        gym.spaces.Box(low=-max_val, high=max_val, dtype=self._dtype))

  def _get_original_observation(self):
    phases = self._robot.timestamp * self._frequencies * 2 * np.pi
    sincos = np.concatenate((np.sin(phases), np.cos(phases)))
    return self._robot.timestamp, sincos

  def get_observation(self) -> np.ndarray:
    delayed_observation = super().get_observation()
    if self._noise_generator:
      if callable(self._noise_generator):
        return self._noise_generator(delayed_observation)
      else:
        return self._noise_generator.add_noise(delayed_observation)

    return delayed_observation