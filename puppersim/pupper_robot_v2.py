import math
import gin
import time
import numpy as np
from pybullet_envs.minitaur.robots import quadruped_base
from pybullet_envs.minitaur.robots import robot_urdf_loader
from pybullet_envs.minitaur.robots import robot_config

from pupper_hardware_interface import interface
from puppersim import pupper_constants
from typing import Dict, Optional, Text, Tuple, Union
from serial.tools import list_ports

@gin.configurable
class PupperRobot(quadruped_base.QuadrupedBase):
  """The Pupper robot interface."""

  def __init__(
      self,
      **kwargs,
  ):
    """Constructs a Pupper robot interface.

    Args:
      **kwargs: The arguments for the parent (Pupper) class.
    """

    self._step_counter = 0

    # The most recent robot state proto received.
    self._robot_state = None

    # Use the instance default worker and the gin configured ip address and
    # port.
    serial_port = next(list_ports.grep(".*ttyACM0.*")).device
    self._hardware_interface = interface.Interface(serial_port)
    time.sleep(0.25)
    self._hardware_interface.set_joint_space_parameters(kp=50.0, kd=5.0, max_current=7.0)
    super().__init__(**kwargs)
    self._clock = time.time


  def _pre_load(self):
    """Import the Pupper specific constants."""
    urdfloader_config = dict(
        pybullet_client=self._pybullet_client,
        urdf_path=pupper_constants.URDF_PATH,
        constrained_base=True,
        # Hang the robot in the air.
        enable_self_collision=True,
        # Makes the simulated robot haning higher from the ground.
        init_base_position=np.array(pupper_constants.INIT_POSITION) + np.array(
            (0, 0, 0.5)),
        init_base_orientation_quaternion=pupper_constants.INIT_ORIENTATION,
        init_joint_angles=pupper_constants.INIT_JOINT_ANGLES,
        joint_offsets=pupper_constants.JOINT_OFFSETS,
        joint_directions=pupper_constants.JOINT_DIRECTIONS,
        motor_names=pupper_constants.MOTOR_NAMES,
        end_effector_names=pupper_constants.END_EFFECTOR_NAMES,
        user_group=pupper_constants.MOTOR_GROUP)
    config_keys = list(urdfloader_config.keys())
    gin_config = gin.config_str()
    for i in range(len(config_keys)):
      if config_keys[i] in gin_config:
        urdfloader_config.pop(config_keys[i])

    self._urdf_loader = robot_urdf_loader.RobotUrdfLoader(
        **urdfloader_config
    )

  def reset(
      self,
      base_position: Optional[Tuple[float]] = None,
      base_orientation_quaternion: Optional[Tuple[float]] = None,
      joint_angles: Optional[Union[Dict[Text, float], Tuple[float]]] = None,
  ):
    """Reset the pupper to its initial states.

    Args:
      base_position: The desired base position. Will use the configured pose in
        gin if None. Does not affect the position of the real robots in general.
      base_orientation_quaternion: The base orientation after resetting. Will
        use the configured values in gin if not specified.
      joint_angles: The desired joint angles after resetting. Will use the
        configured values if None.
    """
    self._robot_state = None
    self._last_action = None

    # self._get_state() will receive a new state proto from Pupper. We also
    # call the self.receive_observation() to update the internal varialbes.
    self._get_state()
    # self.receive_observation()


    joint_angles = [0, 0.6, -1.2] * 4
    super().reset(base_position, base_orientation_quaternion, joint_angles)

    # Receive another state at the end of the reset sequence. Though it is
    # probably not necessary.
    self._get_state()
    self._step_counter = 0
    self._reset_time = self._clock()
 
  def _reset_joint_angles(self,
                          joint_angles: Optional[Union[Tuple[float],
                                                       Dict[Text,
                                                            float]]] = None):
    """Resets the joint pose.

    Args:
      joint_angles: The joint pose if provided. Will use the robot default pose
        from configuration.
    """

    self.set_pose(joint_angles, duration=3.0)

    # Also resets the pose in the animation.
    joint_angles_dict = dict(
        zip(self._motor_id_dict.keys(), joint_angles))
    super()._reset_joint_angles(joint_angles_dict)

    print("exit reset joint angles")

  def set_pose(self, desired_motor_angles, duration):
    """Set the pose of the legs within the given duration."""
    if duration <= 1:
      raise ValueError(
          'The set pose duration of {}s is too short and unsafe.'.format(
              duration))

    assert len(desired_motor_angles) == self.num_motors
    print("enter set pose")

    # Get an initial state.
    self._get_state()
    initial_motor_angles = self.motor_angles

    sequence_start_time = self._clock()
    time_since_start = 0
    while time_since_start < duration:
      progress = time_since_start / duration
      # Use cos to create a soft acceleration/deceleration profile.
      progress = 0.5 * (1 - math.cos(progress * math.pi))
      assert 0 <= progress <= 1
      target_motor_angles = []
      for motor_id, desired_motor_angle in enumerate(
          desired_motor_angles, start=0):
        target_motor_angles.append((1 - progress) *
                                   initial_motor_angles[motor_id] +
                                   progress * desired_motor_angle)

      self.apply_action(target_motor_angles,
                        robot_config.MotorControlMode.POSITION)

      # TODO(tingnan): This sleep is probably not needed?
      time.sleep(0.002)
      time_since_start = self._clock() - sequence_start_time

  def _get_state(self):
    self._hardware_interface.read_incoming_data()
    self._robot_state = self._hardware_interface.robot_state
    self.last_state_time = self._clock()

  def apply_action(self, motor_commands, motor_control_mode=None):
    """Apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    if self._robot_state is None:
      raise AssertionError(
          'No state has been received! Is reset() called before?')

    if motor_control_mode is None:
      motor_control_mode = self._motor_control_mode
    if motor_control_mode is robot_config.MotorControlMode.POSITION:
      assert len(motor_commands) == self.num_motors
      self._hardware_interface.set_actuator_postions(np.array(motor_commands))
    else:
      raise ValueError('{} is not implemented'.format(motor_control_mode))


  def receive_observation(self):
    """Receives the observation from the robot."""
    # States are received within the same send_command cycle, i.e. within
    # self.apply_action(). Here we just use the last received robot state to
    # update the animated robot within PyBullet, and update some internal
    # variables.
#    super()._reset_base_pose(self.base_position,
#                             self.base_orientation_quaternion)

    #joint_angles_dict = dict(zip(self._motor_id_dict.keys(), self.motor_angles))
    #super()._reset_joint_angles(joint_angles_dict)
    self._get_state()

  @property
  def base_position(self):
    """Get the position of pupper's base.

    Kept for compatibility purpose.

    Returns:
      The position of pupper's base.
    """
    #raise NotImplementedError('Not yet implemented!')
    return np.array(pupper_constants.INIT_POSITION)

  @property
  def base_velocity(self):
    """Get the linear velocity of pupper's base.

    TODO(b/172392515): Use mocap to determine base velocity.

    Returns:
      The velocity of pupper's base.
    """
    raise NotImplementedError('Not yet implemented!')

  @property
  def base_roll_pitch_yaw(self):
    """Get pupper's base orientation in euler angle in the world frame.

    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame polluted by noise
      and latency. The unit is rad.
    """
    #raise NotImplementedError('Not yet implemented!')
    return np.asarray([self._robot_state.roll, self._robot_state.pitch, self._robot_state.yaw])

  @property
  def motor_angles(self):
    """Gets the motor angles.

    Returns:
      Motor angles read from Pupper sensors. The motor order is
      [FRONT_RIGHT_ABDUCTION, FRONT_RIGHT_HIP, FRONT_RIGHT_KNEE,
       FRONT_LEFT_ABDUCTION, FRONT_LEFT_HIP, FRONT_LEFT_KNEE,
       REAR_RIGHT_ABDUCTION, REAR_RIGHT_HIP, REAR_RIGHT_KNEE,
       REAR_LEFT_ABDUCTION, REAR_LEFT_HIP, REAR_LEFT_KNEE]
    """
    return np.asarray(self._robot_state.position)

  @property
  def motor_velocities(self):
    """Get the velocities of all motors.

    This function mimicks the noisy sensor reading and adds latency.
    Returns:
      Velocities of all motors polluted by noise and latency.
    """
    return np.asarray(self._robot_state.velocity)

  @property
  def motor_torques(self):
    """Get the amount of torque the motors are exerting.

    Returns:
      A numpy array. Motor torques read from the sensors.
    """
    raise NotImplementedError('Not yet implemented!')


  @property
  def base_orientation_quaternion(self):
    """Get the orientation of pupper's base, represented as quaternion.

    Returns:
      The orientation of robot base as a quaternion.
    """
    raise NotImplementedError('Not yet implemented!')


  @property
  def base_roll_pitch_yaw_rate(self):
    """Get the rate of orientation change of the pupper's base in euler angle.

    Returns:
      rate of (roll, pitch, yaw) change of the robot's base. This is always in
      the local frame of the base.
    """
    return np.asarray([self._robot_state.roll_rate, self._robot_state.pitch_rate, self._robot_state.yaw_rate])

  @property
  def base_acceleration_accelerometer(self):
    """Get the IMU acceleraometer readings."""
    raise NotImplementedError('Not yet implemented!')

  @property
  def base_acceleration(self):
    """Get the base acceleration in the world frame."""
    raise NotImplementedError('Not yet implemented!')

  @classmethod
  def constants(cls):
    del cls
    return pupper_constants

  def convert_leg_pose_to_motor_angles(leg_poses):
    """Convert swing-extend coordinate space to motor angles for a robot type.

    Args:
      leg_poses: A list of leg poses in [abduction, swing, extend] space for all
        4 legs. The order is [abd_0, swing_0, extend_0, abd_1, swing_1,
        extend_1, ...]. Zero swing and zero extend gives a neutral standing
        pose. The conversion is approximate where swing is reflected to hip and
        extend is reflected to both knee and the hip.

    Returns:
      List of 12 motor positions.
    """
    swing_scale = -1.0
    extension_scale = 1.0
    # In this approximate conversion we set hip angle swing + half of the
    # extent and knee angle to extend as rotation.
    # We also scale swing and extend based on some hand-tuned constants.
    multipliers = np.array([1.0, swing_scale, extension_scale] * 4)
    swing_extend_scaled = leg_poses * multipliers
    # Swing is (swing - half of the extension) due to the geometry of the leg.
    extra_swing = swing_extend_scaled * ([0, 0, -0.5] * 4)
    swing_extend_scaled += np.roll(extra_swing, -1)
    motor_angles = list(swing_extend_scaled)
    motor_angles = np.array(PupperRobot.get_neutral_motor_angles()) + motor_angles
    return motor_angles

  def get_neutral_motor_angles():
    ABDUCTION_ANGLE=0
    HIP_ANGLE=0.6
    KNEE_ANGLE=-1.2
    initial_joint_poses = [ABDUCTION_ANGLE,HIP_ANGLE,KNEE_ANGLE]*4
    return initial_joint_poses
