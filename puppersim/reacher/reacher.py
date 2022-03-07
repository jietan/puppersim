import pybullet
import puppersim.data as pd
import time
import math
pybullet.connect(pybullet.GUI)
URDF_PATH = pd.getDataPath()+"/pupper_arm.urdf"
robot_id = pybullet.loadURDF(URDF_PATH, useFixedBase=True)
num_joints = pybullet.getNumJoints(robot_id)
pybullet.setGravity(0, 0, -9.8)
for joint_id in range(pybullet.getNumJoints(robot_id)):
    # Disables the default motors in PyBullet.
    pybullet.setJointMotorControl2(
        bodyIndex=robot_id,
        jointIndex=joint_id,
        controlMode=pybullet.VELOCITY_CONTROL,
        targetVelocity=0,
        force=0)



while True:
    for joint_id in range(pybullet.getNumJoints(robot_id)):
        # Disables the default motors in PyBullet.
        pybullet.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_id,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=math.sin(time.time()))

    pybullet.stepSimulation()
    joint_states = pybullet.getJointStates(robot_id, list(range(num_joints)))
    joint_angles = [joint_data[0] for joint_data in joint_states][0:3]
    joint_velocities = [joint_data[1] for joint_data in joint_states][0:3]
    print("joint angles: ", joint_angles)
    print("joint velocities: ", joint_velocities)
