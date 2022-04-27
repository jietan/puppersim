import random
import math
import time
from sys import platform
import os

import numpy as np
import gym
import pybullet
from pybullet_utils import bullet_client

from pupper_hardware_interface import interface
import puppersim.data as pd
from puppersim.reacher import reacher_kinematics
from serial.tools import list_ports

KP = 6.0
KD = 1.0
MAX_CURRENT = 4.0


class ReacherEnv(gym.Env):
    def __init__(
        self,
        run_on_robot=False,
        render=False,
        torque_control=False,
        torque_penalty=0.0,
        render_meshes=False,
        expand_observation=False,
        leg_index=3,
    ):
        if torque_control:
            self._motor_control = pybullet.TORQUE_CONTROL
            self.action_space = gym.spaces.Box(
                np.array([-0.1, -0.1, -0.1]),
                np.array([0.1, 0.1, 0.1]),
                dtype=np.float32,
            )
        else:
            self._motor_control = pybullet.POSITION_CONTROL
            self.action_space = gym.spaces.Box(
                np.array([-2.0 * math.pi, -1.5 * math.pi, -math.pi]),
                np.array([2.0 * math.pi, 1.5 * math.pi, math.pi]),
                dtype=np.float32,
            )

        self.expand_observation = expand_observation
        self._leg_index = leg_index
        self.torque_penalty = torque_penalty
        self._run_on_robot = run_on_robot

        if self._run_on_robot:
            serial_port = reacher_robot_utils.get_serial_port()
            self._hardware_interface = interface.Interface(serial_port)
            time.sleep(0.25)
            self._hardware_interface.set_joint_space_parameters(
                kp=KP, kd=KD, max_current=MAX_CURRENT
            )
        else:
            if render:
                self._bullet_client = bullet_client.BulletClient(
                    connection_mode=pybullet.GUI
                )
                self._bullet_client.configureDebugVisualizer(
                    self._bullet_client.COV_ENABLE_GUI, 0
                )
                self._bullet_client.resetDebugVisualizerCamera(
                    cameraDistance=0.3,
                    cameraYaw=-134,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 0.1],
                )
            else:
                self._bullet_client = bullet_client.BulletClient(
                    connection_mode=pybullet.DIRECT
                )

        if render_meshes:
            self.urdf_filename = "pupper_arm.urdf"
        else:
            self.urdf_filename = "pupper_arm_no_mesh.urdf"

        obs_shape = self.reset().shape
        # note: don't try to normalize the observation space...
        self.observation_space = gym.spaces.Box(
            low=np.ones(obs_shape) * -np.inf, high=np.ones(obs_shape) * np.inf
        )

    def reset(self, target=None):
        if target is None:
            # new random target
            target_angles = np.random.uniform(-0.5 * math.pi, 0.5 * math.pi, 3)
            self.target = reacher_kinematics.calculate_forward_kinematics_robot(
                target_angles
            )
        else:
            self.target = target

        if self._run_on_robot:
            reacher_robot_utils.blocking_move(
                self._hardware_interface, goal=np.zeros(3), traverse_time=2.0
            )
            obs = self._get_obs_on_robot()
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
                    force=0,
                )
            self._target_visual_shape = self._bullet_client.createVisualShape(
                self._bullet_client.GEOM_SPHERE, radius=0.015
            )
            self._target_visualization = self._bullet_client.createMultiBody(
                baseVisualShapeIndex=self._target_visual_shape, basePosition=self.target
            )
            obs = self._get_obs()
        return obs

    def setTarget(self, target):
        self.target = target

    def calculateInverseKinematics(self, target_pos):
        # compute end effector pos in cartesian cords given angles
        end_effector_link_id = self._get_end_effector_link_id()
        inverse_kinematics = self._bullet_client.calculateInverseKinematics(
            self.robot_id, end_effector_link_id, target_pos
        )

        return inverse_kinematics

    def _apply_actions(self, actions):
        for joint_id, action in zip(range(self.num_joints), actions):
            joint_velocity = self._bullet_client.getJointState(self.robot_id, joint_id)[
                1
            ]
            joint_pos = self._bullet_client.getJointState(self.robot_id, joint_id)[0]
            if self._motor_control == pybullet.POSITION_CONTROL:
                self._bullet_client.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=self._motor_control,
                    targetPosition=action,
                    maxVelocity=1000,
                    positionGain=0.3,
                )
            else:
                self._bullet_client.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_id,
                    controlMode=self._motor_control,
                    force=action,
                )

    def _apply_actions_on_robot(self, actions):
        full_actions = np.zeros([3, 4])
        full_actions[:, self._leg_index] = np.reshape(actions, 3)

        self._hardware_interface.set_joint_space_parameters(
            kp=KP, kd=KD, max_current=MAX_CURRENT
        )
        self._hardware_interface.set_actuator_postions(np.array(full_actions))

    def render(self, *args, **kwargs):
        # here for gym compatability
        pass

    def _get_obs(self):
        joint_states = self._bullet_client.getJointStates(
            self.robot_id, list(range(self.num_joints))
        )
        joint_angles = [joint_data[0] for joint_data in joint_states][0:3]
        joint_velocities = [joint_data[1] for joint_data in joint_states][0:3]

        if self.expand_observation:
            return np.concatenate(
                [
                    np.cos(joint_angles),
                    np.sin(joint_angles),
                    self.target,
                    self._get_vector_from_end_effector_to_goal(),
                    # keep velocities in reasonable range
                    np.array(joint_velocities) / 20.0,
                ]
            )
        else:
            return self.target

    def _get_obs_on_robot(self):
        self._hardware_interface.read_incoming_data()
        self._robot_state = self._hardware_interface.robot_state
        joint_angles = self._robot_state.position[
            self._leg_index * 3 : self._leg_index * 3 + 3
        ]
        joint_velocities = self._robot_state.velocity[
            self._leg_index * 3 : self._leg_index * 3 + 3
        ]

        if self.expand_observation:
            return np.concatenate(
                [
                    np.cos(joint_angles),
                    np.sin(joint_angles),
                    self.target,
                    self._get_vector_from_end_effector_to_goal(),
                    # keep velocities in reasonable range
                    np.array(joint_velocities) / 20.0,
                ]
            )
        else:
            return self.target

    def step(self, actions):
        if self._run_on_robot:
            self._apply_actions_on_robot(actions)
            ob = self._get_obs_on_robot()
        else:
            self._apply_actions(actions)
            ob = self._get_obs()
            self._bullet_client.stepSimulation()

        torques = []
        for joint_id, action in zip(range(self.num_joints), actions):
            if self._motor_control == pybullet.POSITION_CONTROL:
                torques.append(
                    self._bullet_client.getJointState(self.robot_id, joint_id)[3]
                )
            else:
                torques.append(action)
        reward_dist = -np.linalg.norm(self._get_vector_from_end_effector_to_goal())
        reward_ctrl = -self.torque_penalty * np.linalg.norm(torques)
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
            joint_angles = self._robot_state.position[
                self._leg_index * 3 : self._leg_index * 3 + 3
            ]
            end_effector_pos = reacher_kinematics.calculate_forward_kinematics_robot(
                joint_angles
            )
        else:
            end_effector_link_id = self._get_end_effector_link_id()
            end_effector_pos = self._bullet_client.getLinkState(
                bodyUniqueId=self.robot_id,
                linkIndex=end_effector_link_id,
                computeForwardKinematics=1,
            )[0]
        return np.array(end_effector_pos) - np.array(self.target)

    def shutdown(self):
        # TODO: Added this function to attempt to gracefully close
        # the serial connection to the Teensy so that the robot
        # does not jerk, but it doesn't actually work
        self._hardware_interface.serial_handle.close()
