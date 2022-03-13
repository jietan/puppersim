import pybullet
import puppersim.data as pd
import time
import math
import gym
import numpy as np
import random

class ReacherEnv(gym.Env):
    def __init__(self):
        pybullet.connect(pybullet.GUI)

    def reset(self):
        pybullet.resetSimulation()
        URDF_PATH = pd.getDataPath()+"/pupper_arm.urdf"
        self.robot_id = pybullet.loadURDF(URDF_PATH, useFixedBase=True)
        pybullet.setGravity(0, 0, -9.8)
        self.num_joints = pybullet.getNumJoints(self.robot_id)
        for joint_id in range(self.num_joints):
            # Disables the default motors in PyBullet.
            pybullet.setJointMotorControl2(
                bodyIndex=self.robot_id,
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
        inverse_kinematics = pybullet.calculateInverseKinematics(self.robot_id, end_effector_link_id, target_pos)

        return inverse_kinematics


    def step(self, actions):
        for joint_id, action in zip(range(self.num_joints), actions):
            # Disables the default motors in PyBullet.
            pybullet.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_id,
                controlMode=pybullet.POSITION_CONTROL,
                targetPosition=action)
        
        ob = self._get_obs()

        reward_dist = -np.linalg.norm(self._get_vector_from_end_effector_to_goal())
        reward_ctrl = 0
        reward = reward_dist + reward_ctrl

        done = False

        pybullet.stepSimulation()
        return ob, reward, done, {}

    def _get_end_effector_link_id(self):
        for joint_id in range(self.num_joints):
            joint_name = pybullet.getJointInfo(self.robot_id, joint_id)[1]
            if joint_name.decode("UTF-8") == "leftFrontToe":
                return joint_id
        raise ValueError("leftFrontToe not found")

    def _get_vector_from_end_effector_to_goal(self):
        end_effector_link_id = self._get_end_effector_link_id()
        end_effector_pos = pybullet.getLinkState(bodyUniqueId=self.robot_id, linkIndex=end_effector_link_id, computeForwardKinematics=1)[0]
        return np.array(end_effector_pos) - np.array(self.target)

    def _get_obs(self):
        joint_states = pybullet.getJointStates(self.robot_id, list(range(self.num_joints)))
        joint_angles = [joint_data[0] for joint_data in joint_states][0:3]
        joint_velocities = [joint_data[1] for joint_data in joint_states][0:3]
        return np.concatenate(
            [
                np.cos(joint_angles),
                np.sin(joint_angles),
                self.target,
                joint_velocities,
                self._get_vector_from_end_effector_to_goal(),
            ]
        )
