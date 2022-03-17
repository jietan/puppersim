import pybullet
import puppersim.data as pd
import time
import math
import gym
import numpy as np
import random
from pupper_hardware_interface import interface
from serial.tools import list_ports

KP = 16.0
KD = 2.0
MAX_CURRENT = 7.0


class ReacherEnv(gym.Env):

  def __init__(self, run_on_robot=False):
    self._run_on_robot = run_on_robot
    if self._run_on_robot:
      # serial_port = next(list_ports.grep(".*ttyACM0.*")).device
      serial_port = next(list_ports.grep("usbmodem")).device
      self._hardware_interface = interface.Interface(serial_port)
      time.sleep(0.25)
      self._hardware_interface.set_joint_space_parameters(
          kp=KP, kd=KD, max_current=MAX_CURRENT)
    else:
      pybullet.connect(pybullet.GUI)

  def reset(self):
    if self._run_on_robot:
      pass
    else:
      pybullet.resetSimulation()
      URDF_PATH = pd.getDataPath() + "/pupper_arm.urdf"
      self.robot_id = pybullet.loadURDF(URDF_PATH, useFixedBase=True)
      pybullet.setGravity(0, 0, -9.8)
      self.num_joints = pybullet.getNumJoints(self.robot_id)
      for joint_id in range(self.num_joints):
        # Disables the default motors in PyBullet.
        pybullet.setJointMotorControl2(bodyIndex=self.robot_id,
                                       jointIndex=joint_id,
                                       controlMode=pybullet.VELOCITY_CONTROL,
                                       targetVelocity=0,
                                       force=0)
    self.target = np.random.uniform(0.05, 0.1, 3)

  def setTarget(self, target):
    self.target = target

  def calculateInverseKinematics(self, target_pos):
    # compute end effector pos in cartesian cords given angles
    end_effector_link_id = self._get_end_effector_link_id()
    inverse_kinematics = pybullet.calculateInverseKinematics(
        self.robot_id, end_effector_link_id, target_pos)

    return inverse_kinematics

  def _apply_actions(self, actions):
    for joint_id, action in zip(range(self.num_joints), actions):
      # Disables the default motors in PyBullet.
      pybullet.setJointMotorControl2(bodyIndex=self.robot_id,
                                     jointIndex=joint_id,
                                     controlMode=pybullet.POSITION_CONTROL,
                                     targetPosition=action)

  def _apply_actions_on_robot(self, actions):
    full_actions = np.zeros([3, 4])
    full_actions[:, 2] = np.array(actions)
    self._hardware_interface.set_joint_space_parameters(kp=KP,
                                                        kd=KD,
                                                        max_current=MAX_CURRENT)
    self._hardware_interface.set_actuator_postions(np.array(full_actions))

  def _get_obs(self):
    joint_states = pybullet.getJointStates(self.robot_id,
                                           list(range(self.num_joints)))
    joint_angles = [joint_data[0] for joint_data in joint_states][0:3]
    joint_velocities = [joint_data[1] for joint_data in joint_states][0:3]
    return np.concatenate([
        np.cos(joint_angles),
        np.sin(joint_angles),
        self.target,
        joint_velocities,
        self._get_vector_from_end_effector_to_goal(),
    ])

  def _get_obs_on_robot(self):
    self._hardware_interface.read_incoming_data()
    self._robot_state = self._hardware_interface.robot_state

    joint_angles = self._robot_state.position[6:9]
    joint_velocities = self._robot_state.velocity[6:9]
    return np.concatenate([
        np.cos(joint_angles),
        np.sin(joint_angles),
        self.target,
        joint_velocities,
        self._get_vector_from_end_effector_to_goal(),
    ])

  def step(self, actions):
    if self._run_on_robot:
      self._apply_actions_on_robot(actions)
      ob = self._get_obs_on_robot()
    else:
      self._apply_actions(actions)
      ob = self._get_obs()
      pybullet.stepSimulation()

    reward_dist = -np.linalg.norm(self._get_vector_from_end_effector_to_goal())
    reward_ctrl = 0
    reward = reward_dist + reward_ctrl

    done = False

    return ob, reward, done, {}

  def _get_end_effector_link_id(self):
    for joint_id in range(self.num_joints):
      joint_name = pybullet.getJointInfo(self.robot_id, joint_id)[1]
      if joint_name.decode("UTF-8") == "leftFrontToe":
        return joint_id
    raise ValueError("leftFrontToe not found")

  def _forward_kinematics(self, joint_angles):
    #TODO (nathan) add this
    return np.array([0, 0, 0])

  def _get_vector_from_end_effector_to_goal(self):
    if self._run_on_robot:
      joint_angles = self._robot_state.position[6:9]
      end_effector_pos = self._forward_kinematics(joint_angles)
    else:
      end_effector_link_id = self._get_end_effector_link_id()
      end_effector_pos = pybullet.getLinkState(bodyUniqueId=self.robot_id,
                                               linkIndex=end_effector_link_id,
                                               computeForwardKinematics=1)[0]
    return np.array(end_effector_pos) - np.array(self.target)

  def shutdown(self):
    # TODO: Added this function to attempt to gracefully close
    # the serial connection to the Teensy so that the robot
    # does not jerk, but it doesn't actually work
    self._hardware_interface.serial_handle.close()