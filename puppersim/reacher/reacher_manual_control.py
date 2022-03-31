import pybullet as p
import puppersim.data as pd
import time
import math
import numpy as np
from absl import app
from absl import flags
import copy
from pupper_hardware_interface import interface
from serial.tools import list_ports

flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
FLAGS = flags.FLAGS

run_on_robot = True

import pybullet_data

KP = 4.0
KD = 4.0
MAX_CURRENT = 4.0

HIP_OFFSET = 0.0335
L1 = 0.08
L2 = 0.11

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
URDF_PATH = pd.getDataPath() + "/pupper_arm.urdf"
reacher = p.loadURDF(URDF_PATH, useFixedBase=True)

gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
jointIds = []
paramIds = []

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(reacher, -1, linearDamping=0, angularDamping=0)

for j in range(p.getNumJoints(reacher)):
  p.changeDynamics(reacher, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(reacher, j)
  #print(info)
  jointName = info[1]
  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    jointIds.append(j)
    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

if run_on_robot:
  serial_port = next(list_ports.grep("usbmodem")).device
  hardware_interface = interface.Interface(serial_port)
  time.sleep(0.25)
  hardware_interface.set_joint_space_parameters(
      kp=KP, kd=KD, max_current=MAX_CURRENT)


def calculate_forward_kinematics_robot(joint_angles):
  # compute end effector pos in cartesian cords given angles

  x1 = L1 * math.sin(joint_angles[1])
  z1 = L1 * math.cos(joint_angles[1])

  x2 = L2 * math.sin(joint_angles[1] + joint_angles[2])
  z2 = L2 * math.cos(joint_angles[1] + joint_angles[2])

  foot_pos = np.array([[HIP_OFFSET],
                      [x1 + x2], 
                      [z1 + z2]
                      ])

  rot_mat = np.array([[math.cos(-joint_angles[0]), -math.sin(-joint_angles[0]), 0],
                      [math.sin(-joint_angles[0]), math.cos(-joint_angles[0]), 0],
                      [0, 0, 1]
                      ])  

  end_effector_pos = np.matmul(rot_mat, foot_pos)

  xyz = np.transpose(end_effector_pos)

  return xyz[0]


p.setRealTimeSimulation(1)
counter = 0
while (1):
  counter += 1
  joint_angles = [0, 0, 0]
  p.setGravity(0, 0, p.readUserDebugParameter(gravId))
  for i in range(len(paramIds)):
    c = paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    joint_angles[i] = targetPos
    p.setJointMotorControl2(reacher, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)

  if run_on_robot:
    full_actions = np.zeros([3, 4])
    full_actions[:, 2] = np.reshape(joint_angles, 3)
    hardware_interface.set_joint_space_parameters(kp=KP,
                                                        kd=KD,
                                                        max_current=MAX_CURRENT)
    hardware_interface.set_actuator_postions(np.array(full_actions))

  if counter % 100 == 0:
    print(calculate_forward_kinematics_robot(joint_angles))
  time.sleep(0.01)

