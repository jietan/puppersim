# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Add the new Pupper robot."""
import gin
import gym
import numpy as np
from puppersim import pupper_constants
from pybullet_envs.minitaur.robots import quadruped_base
from pybullet_envs.minitaur.robots import robot_urdf_loader
from pybullet_envs.minitaur.robots import robot_config

@gin.configurable
class Pupper(quadruped_base.QuadrupedBase):
  """The Pupper class that simulates the quadruped from Unitree."""

  def _pre_load(self):
    """Import the Pupper specific constants.
    """
    self._urdf_loader = robot_urdf_loader.RobotUrdfLoader(
        pybullet_client=self._pybullet_client,
        urdf_path=pupper_constants.URDF_PATH,
        enable_self_collision=False ,
        init_base_position=pupper_constants.INIT_POSITION,
        init_base_orientation_quaternion=pupper_constants.INIT_ORIENTATION,
        init_joint_angles=pupper_constants.INIT_JOINT_ANGLES,
        joint_offsets=pupper_constants.JOINT_OFFSETS,
        joint_directions=pupper_constants.JOINT_DIRECTIONS,
        motor_names=pupper_constants.MOTOR_NAMES,
        end_effector_names=pupper_constants.END_EFFECTOR_NAMES,
        user_group=pupper_constants.MOTOR_GROUP,
    )
  def _on_load(self):
    self._joint_id_dict = self._urdf_loader.get_joint_id_dict()
    for joint_id in self._joint_id_dict.values():
      # set a default friction force for all motors in PyBullet.
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self._urdf_loader.robot_id,
          jointIndex=joint_id,
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=pupper_constants.JOINT_FRICTION_FORCE)

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
    motor_angles = np.array(Pupper.get_neutral_motor_angles()) + motor_angles
    return motor_angles

  def get_neutral_motor_angles():
    ABDUCTION_ANGLE=0
    HIP_ANGLE=0.6
    KNEE_ANGLE=-1.2
    initial_joint_poses = [ABDUCTION_ANGLE,HIP_ANGLE,KNEE_ANGLE]*4
    return initial_joint_poses

  @property
  def base_roll_pitch_yaw(self):
    return [self._base_roll_pitch_yaw[1], -self._base_roll_pitch_yaw[0], self._base_roll_pitch_yaw[2]]

  @property
  def base_roll_pitch_yaw_rate(self):
    return [self._base_roll_pitch_yaw_rate[1], -self._base_roll_pitch_yaw_rate[0], self._base_roll_pitch_yaw_rate[2]]

  @classmethod
  def get_constants(cls):
    del cls
    return pupper_constants
