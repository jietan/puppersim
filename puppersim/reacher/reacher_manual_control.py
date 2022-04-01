from puppersim.reacher import reacher_kinematics
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
from sys import platform

flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
FLAGS = flags.FLAGS
import pybullet_data

KP = 10.0
KD = 2.0
MAX_CURRENT = 16.0

HIP_OFFSET = 0.0335
L1 = 0.08
L2 = 0.11

def main(argv):
  run_on_robot = FLAGS.run_on_robot
  p.connect(p.GUI)
  p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  URDF_PATH = pd.getDataPath() + "/pupper_arm.urdf"
  reacher = p.loadURDF(URDF_PATH, useFixedBase=True)

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
      paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -math.pi, math.pi, 0))

  if run_on_robot:
    if platform == "linux" or platform == "linux2":
      serial_port = next(list_ports.grep(".*ttyACM0.*")).device
    elif platform == "darwin":
      serial_port = next(list_ports.grep("usbmodem")).device
    hardware_interface = interface.Interface(serial_port)
    time.sleep(0.25)
    hardware_interface.set_joint_space_parameters(
        kp=KP, kd=KD, max_current=MAX_CURRENT)

  p.setRealTimeSimulation(1)
  counter = 0
  while (1):
    counter += 1
    joint_angles = [0, 0, 0]
    for i in range(len(paramIds)):
      c = paramIds[i]
      targetPos = p.readUserDebugParameter(c)
      joint_angles[i] = targetPos
      p.setJointMotorControl2(reacher, jointIds[i], p.POSITION_CONTROL, targetPos, force=2 * 240.)

    if run_on_robot:
      full_actions = np.zeros([3, 4])
      full_actions[:, 2] = np.reshape(joint_angles, 3)
      hardware_interface.set_joint_space_parameters(kp=KP,
                                                    kd=KD,
                                                    max_current=MAX_CURRENT)
      hardware_interface.set_actuator_postions(np.array(full_actions))

    if counter % 5 == 0:
      print(reacher_kinematics.calculate_forward_kinematics_robot(joint_angles))
    time.sleep(0.01)

app.run(main)