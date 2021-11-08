# Lint as: python3
"""A generic PD motor model."""

from typing import Tuple, Union
import gin
import numpy as np

from pybullet_envs.minitaur.robots import robot_config
from pybullet_envs.minitaur.robots import time_ordered_buffer

def _convert_to_np_array(inputs: Union[float, Tuple[float], np.ndarray], dim):
  """Converts the inputs to a numpy array.

  Args:
    inputs: The input scalar or array.
    dim: The dimension of the converted numpy array.

  Returns:
    The converted numpy array.

  Raises:
    ValueError: If the inputs is an array whose dimension does not match the
    provied dimension.
  """
  outputs = None
  if isinstance(inputs, (tuple, np.ndarray)):
    outputs = np.array(inputs)
  else:
    outputs = np.full(dim, inputs)

  if len(outputs) != dim:
    raise ValueError("The inputs array has a different dimension {}"
                     " than provided, which is {}.".format(len(outputs), dim))

  return outputs


class FirstOrderFilter:
  """
  First order complementary filter.

  Gain is unity until time_constant at which point it is -20dB/dec.
  """
  def __init__(self, time_constant: float, sampling_time: float):
    """Initializes the first order filter.

    Requires that filter is called at regular intervals specified by sampling_time

    Computes the complementary factor as, 
    alpha = sampling_time / (time_constant + sampling_time),
    which is valid for time_constant >> sampling_time.
    
    Args:
      time_constant: time constant [s]
      sampling_time: sampling time [s]
    """
    self.alpha = sampling_time / (time_constant + sampling_time)
    self.state = None

  def __call__(self, input: Union[np.ndarray, float]):
    """
    Updates the filter and returns the new filtered value.
    
    Accepts floats and np.arrays but you cannot switch between them at runtime.
    
    Args:
      input: input
    
    Returns:
      Filtered output
    """
    if self.state is not None:
      self.state = self.alpha * input + (1 - self.alpha) * self.state
    else:
      self.state = input
    return self.state 

@gin.configurable
class PupperMotorModel(object):
  """A simple motor model that supports proportional and derivative control.

    When in POSITION mode, the torque is calculated according to the difference
    between current and desired joint angle, as well as the joint velocity
    differences. For more information about PD control, please refer to:
    https://en.wikipedia.org/wiki/PID_controller.

    The model supports a HYBRID mode in which each motor command can be a tuple
    (desired_motor_angle, position_gain, desired_motor_velocity, velocity_gain,
    torque).
  """

  def __init__(
      self,
      num_motors: int,
      sampling_time: float = 0.001,
      pd_latency: float = 0,
      motor_control_mode=robot_config.MotorControlMode.POSITION,
      kp: Union[float, Tuple[float], np.ndarray] = 60,
      kd: Union[float, Tuple[float], np.ndarray] = 1,
      strength_ratios: Union[float, Tuple[float], np.ndarray] = 1,
      torque_lower_limits: Union[float, Tuple[float], np.ndarray] = None,
      torque_upper_limits: Union[float, Tuple[float], np.ndarray] = None,
      velocity_filter_time_constant: float = 0.0,
      torque_time_constant: float = 0.0, # BUG: torque filter causes instability
      motor_damping: float = 0.0,
      motor_torque_dependent_friction: float = 0.0,    
  ):
    """Initializes the class.

    Args:
      num_motors: The number of motors for parallel computation.
      sampling_time: Interval between model updates [s].
      pd_latency: Simulates the motor controller's latency in reading motor
        angles and velocities.
      motor_control_mode: Can be POSITION, TORQUE, or HYBRID. In POSITION
        control mode, the PD formula is used to track a desired position and a
        zero desired velocity. In TORQUE control mode, we assume a pass through
        of the provided torques. In HYBRID control mode, the users need to
        provie (desired_position, position_gain, desired_velocity,
        velocity_gain, feedfoward_torque) for each motor.
      kp: The default position gains for motors.
      kd: The default velocity gains for motors.
      strength_ratios: The scaling ratio for motor torque outputs. This can be
        useful for quick debugging when sim-to-real gap is observed in the
        actuator behavior.
      torque_lower_limits: The lower bounds for torque outputs.
      torque_upper_limits: The upper bounds for torque outputs. The output
        torques will be clipped by the lower and upper bounds.
      velocity_filter_time_constant: Time constant for the velocity filter.
      torque_time_constant: Time constant for the actuator's transfer 
        function between requested torque and actual torque.
      motor_damping: Damping in [Nm/(rad/s)] of the motor output. Note that
        coulomb friction is handled by pybullet directly
      motor_torque_dependent_friction: Coulomb friction per Nm of motor torque, unitless.

    Raises:
      ValueError: If the number of motors provided is negative or zero.
    """
    if num_motors <= 0:
      raise ValueError(
          "Number of motors must be positive, not {}".format(num_motors))
    self._num_motors = num_motors
    self._zero_array = np.full(num_motors, 0)
    self._pd_latency = pd_latency
    self.set_motor_gains(kp, kd)
    self.set_strength_ratios(strength_ratios)
    self._torque_lower_limits = None
    if torque_lower_limits:
      self._torque_lower_limits = _convert_to_np_array(torque_lower_limits,
                                                       self._num_motors)

    self._torque_upper_limits = None
    if torque_upper_limits:
      self._torque_upper_limits = _convert_to_np_array(torque_upper_limits,
                                                       self._num_motors)
    self._motor_control_mode = motor_control_mode

    self._velocity_filter = FirstOrderFilter(time_constant=velocity_filter_time_constant, 
                                             sampling_time=sampling_time)
    self._torque_filter = FirstOrderFilter(time_constant=torque_time_constant,
                                           sampling_time=sampling_time)

    # Used for modeling continuous-time actuator dynamics
    self._physical_velocity_filter = FirstOrderFilter(time_constant=2*sampling_time,
                                                      sampling_time=sampling_time)

    self._motor_damping = motor_damping
    self._motor_torque_dependent_friction = motor_torque_dependent_friction

    self._previous_true_motor_velocity = 0.0

    # The history buffer is used to simulate the pd latency effect.
    # TODO(b/157786642): remove hacks on duplicate timestep once the sim clock
    # is fixed.
    self._observation_buffer = time_ordered_buffer.TimeOrderedBuffer(
        max_buffer_timespan=pd_latency,
        error_on_duplicate_timestamp=False,
        replace_value_on_duplicate_timestamp=True)

  def set_strength_ratios(
      self,
      strength_ratios: Union[float, Tuple[float], np.ndarray],
  ):
    """Sets the strength of each motor relative to the default value.

    Args:
      strength_ratios: The relative strength of motor output, ranging from [0,
        1] inclusive.
    """
    self._strength_ratios = np.clip(
        _convert_to_np_array(strength_ratios, self._num_motors), 0, 1)

  def set_motor_gains(
      self,
      kp: Union[float, Tuple[float], np.ndarray],
      kd: Union[float, Tuple[float], np.ndarray],
  ):
    """Sets the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: Proportional gain of the motors.
      kd: Derivative gain of the motors.
    """
    self._kp = _convert_to_np_array(kp, self._num_motors)
    self._kd = _convert_to_np_array(kd, self._num_motors)

  def get_motor_gains(self):
    """Get the PD gains of all motors.

    Returns:
      Proportional and derivative gain of the motors.
    """
    return self._kp, self._kd

  def reset(self):
    self._observation_buffer.reset()

  def update(self, timestamp, true_motor_positions: np.ndarray,
             true_motor_velocities: np.ndarray):
    # Filter the motor_velocities to mimick the C610's velocity filter dynamics
    filtered_motor_velocities = self._velocity_filter(true_motor_velocities)

    # Keep track of the last true motor velocity in order to model actuator dynamics
    self._previous_true_motor_velocity = self._physical_velocity_filter(true_motor_velocities)

    # Push these to the buffer
    self._observation_buffer.add(timestamp,
                                 (true_motor_positions, filtered_motor_velocities))

  def get_motor_torques(
      self,
      motor_commands: np.ndarray,
      motor_control_mode=None) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the motor torques.

    Args:
      motor_commands: The desired motor angle if the motor is in position
        control mode. The pwm signal if the motor is in torque control mode.
      motor_control_mode: A MotorControlMode enum.

    Returns:
      observed_torque: The torque observed. This emulates the limitations in
      torque measurement, which is generally obtained from current estimations.
      actual_torque: The torque that needs to be applied to the motor.

    Raises:
      NotImplementedError if the motor_control_mode is not supported.

    """
    if not motor_control_mode:
      motor_control_mode = self._motor_control_mode

    motor_torques = None

    if motor_control_mode is robot_config.MotorControlMode.TORQUE:
      motor_torques = motor_commands

    if motor_control_mode is robot_config.MotorControlMode.POSITION:
      motor_torques = self._compute_pd_torques(
          desired_motor_angles=motor_commands,
          kp=self._kp,
          desired_motor_velocities=self._zero_array,
          kd=self._kd)
      
    if motor_torques is None:
      raise ValueError(
          "{} is not a supported motor control mode".format(motor_control_mode))

    # Apply the output filter to model actuator dynamics
    # BUG: Causes big instability in the sim
    # motor_torques = self._torque_filter(motor_torques)

    # Hard-code torque limits until the torque limit bug is fixed
    motor_torques = np.clip(motor_torques, -1.7, 1.7)

    # Apply motor damping and friction
    motor_torques -= (np.sign(self._previous_true_motor_velocity) *
                      self._motor_torque_dependent_friction *
                      motor_torques)
    motor_torques -= self._previous_true_motor_velocity * self._motor_damping

    # Rescale and clip the motor torques as needed.
    motor_torques = self._strength_ratios * motor_torques
    if (self._torque_lower_limits is not None or
        self._torque_upper_limits is not None):
      motor_torques = np.clip(motor_torques, self._torque_lower_limits,
                              self._torque_upper_limits)

    return motor_torques, motor_torques

  def get_motor_states(self, latency=None):
    """Computes observation of motor angle and velocity under latency."""
    if latency is None:
      latency = self._pd_latency
    buffer = self._observation_buffer.get_delayed_value(latency)
    angle_vel_t0 = buffer.value_0
    angle_vel_t1 = buffer.value_1
    coeff = buffer.coeff

    pos_idx = 0
    motor_angles = angle_vel_t0[pos_idx] * (
        1 - coeff) + coeff * angle_vel_t1[pos_idx]
    vel_idx = 1
    motor_velocities = angle_vel_t0[vel_idx] * (
        1 - coeff) + coeff * angle_vel_t1[vel_idx]
    return motor_angles, motor_velocities

  def _compute_pd_torques(
      self,
      desired_motor_angles: np.ndarray,
      kp: np.ndarray,
      desired_motor_velocities,
      kd: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the pd torques.

    Args:
      desired_motor_angles: The motor angles to track.
      kp: The position gains.
      desired_motor_velocities: The motor velocities to track.
      kd: The velocity gains.

    Returns:
      The computed motor torques.
    """
    motor_angles, motor_velocities = self.get_motor_states()
    motor_torques = -kp * (motor_angles - desired_motor_angles) - kd * (
        motor_velocities - desired_motor_velocities)

    return motor_torques
