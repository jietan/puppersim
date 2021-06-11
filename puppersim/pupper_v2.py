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


  def _build_action_space(self):
     """Builds the action space of the robot using the motor limits."""
     if self._motor_control_mode == robot_config.MotorControlMode.POSITION:
      self._action_space = gym.spaces.Box(
          low=np.array([-0.18,0.1,-2.3]*4),#-0.0,#self._motor_limits.angle_lower_limits,
          high=np.array([0.18,0.7,-0.6]*4),#0.001  ,#self._motor_limits.angle_upper_limits,
          shape=(self._num_motors,),
          dtype=np.float32)  # TODO(b/159160184) Make dtype configurable.
      self._action_names = tuple(
          "POSITION_{}".format(motor) for motor in self._motor_id_dict.keys())
     else:
        raise NotImplementedError("Not yet implemented!")
  def get_neutral_motor_angles():
    ABDUCTION_ANGLE=0
    HIP_ANGLE=0.6
    KNEE_ANGLE=-1.2
    initial_joint_poses = [ABDUCTION_ANGLE,HIP_ANGLE,KNEE_ANGLE]*4
    return initial_joint_poses
    

  @classmethod
  def get_constants(cls):
    del cls
    return pupper_constants
