import pybullet
import puppersim.data as pd
from pybullet_utils import bullet_client
from puppersim.reacher import reacher_kinematics
from puppersim.reacher import reacher_robot_utils
import time
import math
import gym
import numpy as np
import random
from pupper_hardware_interface import interface
from serial.tools import list_ports
import os

KP = 6.0
KD = 1.0
MAX_CURRENT = 4.0


class ReacherEnv(gym.Env):

  def __init__(
      self,
      run_on_robot=False,
      render=False,
      render_meshes=False,
      leg_index=3,
  ):
    self.action_space = gym.spaces.Box(
        np.array([-2 * math.pi, -1.5 * math.pi, -1.0 * math.pi]),
        np.array([2 * math.pi, 1.5 * math.pi, 1.0 * math.pi]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        # np.array([-1, -1, -1, -1, -1, -1, 0.05, 0.05, 0.05, -0.3, -0.3, -0.3]),
        # np.array([1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3]),
        # observation space range for target
        np.array([-0.1, -0.1, 0.05]),
        np.array([0.1, 0.1, 0.15]),
        dtype=np.float32)
    self._leg_index = leg_index

    self.target = np.array([0, 0, 0.1])

    self._run_on_robot = run_on_robot
    if self._run_on_robot:
      serial_port = reacher_robot_utils.get_serial_port()
      self._hardware_interface = interface.Interface(serial_port)
      time.sleep(0.25)
      self._hardware_interface.set_joint_space_parameters(
          kp=KP, kd=KD, max_current=MAX_CURRENT)
    else:
      if render:
        self._bullet_client = bullet_client.BulletClient(
            connection_mode=pybullet.GUI)
        self._bullet_client.configureDebugVisualizer(
            self._bullet_client.COV_ENABLE_GUI, 0)
        self._bullet_client.resetDebugVisualizerCamera(
            cameraDistance=0.3,
            cameraYaw=46,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.1])
      else:
        self._bullet_client = bullet_client.BulletClient(
            connection_mode=pybullet.DIRECT)

    if render_meshes:
      self.urdf_filename = "pupper_arm.urdf"
    else:
      self.urdf_filename = "pupper_arm_no_mesh.urdf"

  def reset(self, target=None):
    self.target = target if target is not None else np.array([0.00, 0.00, 0.1])

    if self._run_on_robot:
      reacher_robot_utils.blocking_move(self._hardware_interface,
                                        goal=np.zeros(3),
                                        traverse_time=2.0)
      obs =  self._get_obs_on_robot()
    else:
      self._bullet_client.resetSimulation()
      URDF_PATH = os.path.join(pd.getDataPath(), self.urdf_filename)
      self.robot_id = self._bullet_client.loadURDF(URDF_PATH, useFixedBase=True)
      self._bullet_client.setGravity(0, 0, -9.8)
      self.num_joints = self._bullet_client.getNumJoints(self.robot_id)
      for joint_id in range(self.num_joints):
        # Disables the default motors in PyBullet.
        self._bullet_client.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=joint_id,
            controlMode=self._bullet_client.POSITION_CONTROL,
            targetVelocity=0,
            force=0)

      # self.target = np.random.uniform(0.05, 0.10, 3)
      # Target position: xy range of -0.1 to 0.1. z range of 0.05 to 0.15.
      # self.target = np.concatenate([np.random.uniform(-0.1, 0.1, 2), np.random.uniform(0.05, 0.15, 1)])

      # possible_targets = []
      # possible_targets.append(np.array([-0.07, -0.07, 0.07]))
      # possible_targets.append(np.array([0.07, 0.07, 0.07]))
      # possible_targets.append(np.array([-0.07, 0.07, 0.07]))
      # possible_targets.append(np.array([0.07, -0.07, 0.07]))
      # self.target = random.choice(possible_targets)

      # target_angles = np.random.uniform(-0.5*math.pi, 0.5*math.pi, 3)
      # self.target = reacher_kinematics.calculate_forward_kinematics_robot(target_angles)

      self._target_visual_shape = self._bullet_client.createVisualShape(
          self._bullet_client.GEOM_SPHERE, radius=0.015)
      self._target_visualization = self._bullet_client.createMultiBody(
          baseVisualShapeIndex=self._target_visual_shape,
          basePosition=self.target)

      obs = self._get_obs()

    return obs

  def setTarget(self, target):
    self.target = target

  def calculateInverseKinematics(self, target_pos):
    # compute end effector pos in cartesian cords given angles
    end_effector_link_id = self._get_end_effector_link_id()
    inverse_kinematics = self._bullet_client.calculateInverseKinematics(
        self.robot_id, end_effector_link_id, target_pos)

    return inverse_kinematics

  def _apply_actions(self, actions):
    for joint_id, action in zip(range(self.num_joints), actions):
      # Disables the default motors in PyBullet.
      self._bullet_client.setJointMotorControl2(
          bodyIndex=self.robot_id,
          jointIndex=joint_id,
          controlMode=pybullet.POSITION_CONTROL,
          targetPosition=action,
          maxVelocity=1000,
          positionGain=0.3)

  def _apply_actions_on_robot(self, actions):
    full_actions = np.zeros([3, 4])
    full_actions[:, self._leg_index] = np.reshape(actions, 3)

    self._hardware_interface.set_joint_space_parameters(kp=KP,
                                                        kd=KD,
                                                        max_current=MAX_CURRENT)
    self._hardware_interface.set_actuator_postions(np.array(full_actions))

  def _get_obs(self):
    joint_states = self._bullet_client.getJointStates(
        self.robot_id, list(range(self.num_joints)))
    joint_angles = [joint_data[0] for joint_data in joint_states][0:3]
    joint_velocities = [joint_data[1] for joint_data in joint_states][0:3]
    return np.concatenate([
        # np.cos(joint_angles),
        # np.sin(joint_angles),
        self.target,
        # joint_velocities,
        # self._get_vector_from_end_effector_to_goal(),
    ])

  def _get_obs_on_robot(self):
    self._hardware_interface.read_incoming_data()
    self._robot_state = self._hardware_interface.robot_state
    joint_angles = self._robot_state.position[self._leg_index *
                                              3:self._leg_index * 3 + 3]
    joint_velocities = self._robot_state.velocity[self._leg_index *
                                                  3:self._leg_index * 3 + 3]
    np.set_printoptions(precision=2)
    return np.concatenate([
        # np.cos(joint_angles),
        # np.sin(joint_angles),
        self.target,
        # joint_velocities,
        # self._get_vector_from_end_effector_to_goal(),
    ])

  def step(self, actions):

    if self._run_on_robot:
      self._apply_actions_on_robot(actions)
      ob = self._get_obs_on_robot()
    else:
      self._apply_actions(actions)
      ob = self._get_obs()
      self._bullet_client.stepSimulation()

    reward_dist = -np.linalg.norm(
        self._get_vector_from_end_effector_to_goal())**2
    reward_ctrl = 0
    reward = reward_dist + reward_ctrl

    done = False

    return ob, reward, done, {}

  def _get_end_effector_link_id(self):
    for joint_id in range(self.num_joints):
      joint_name = self._bullet_client.getJointInfo(self.robot_id, joint_id)[1]
      if joint_name.decode("UTF-8") == "leftFrontToe":
        return joint_id
    raise ValueError("leftFrontToe not found")

  def _get_vector_from_end_effector_to_goal(self):
    if self._run_on_robot:
      joint_angles = self._robot_state.position[self._leg_index*3: self._leg_index*3+3]
      end_effector_pos = reacher_kinematics.calculate_forward_kinematics_robot(
          joint_angles)
    else:
      end_effector_link_id = self._get_end_effector_link_id()
      end_effector_pos = self._bullet_client.getLinkState(
          bodyUniqueId=self.robot_id,
          linkIndex=end_effector_link_id,
          computeForwardKinematics=1)[0]
      # print("end effector: ", end_effector_pos)
    return np.array(end_effector_pos) - np.array(self.target)

  def shutdown(self):
    # TODO: Added this function to attempt to gracefully close
    # the serial connection to the Teensy so that the robot
    # does not jerk, but it doesn't actually work
    self._hardware_interface.serial_handle.close()
