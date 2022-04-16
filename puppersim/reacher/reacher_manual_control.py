from puppersim.reacher import reacher_kinematics
from puppersim.reacher import reacher_robot_utils
import pybullet as p
import puppersim.data as pd
import time
import math
import numpy as np
from absl import app
from absl import flags
import copy
from pupper_hardware_interface import interface
from sys import platform

flags.DEFINE_bool("run_on_robot", False,
                  "Whether to run on robot or in simulation.")
FLAGS = flags.FLAGS
import pybullet_data

KP = 5.0  # Amps/rad
KD = 2.0  # Amps/(rad/s)
MAX_CURRENT = 3.0  # Amps

UPDATE_DT = 0.01  # seconds

HIP_OFFSET = 0.0335  # meters
L1 = 0.08  # meters
L2 = 0.11  # meters


def load_reacher():
  p.connect(p.GUI)
  p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  p.resetDebugVisualizerCamera(cameraDistance=0.3,
                               cameraYaw=-134,
                               cameraPitch=-30,
                               cameraTargetPosition=[0, 0, 0.1])

  URDF_PATH = pd.getDataPath() + "/pupper_arm.urdf"
  return p.loadURDF(URDF_PATH, useFixedBase=True)


def main(argv):
  run_on_robot = FLAGS.run_on_robot
  reacher = load_reacher()
  target_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.015)
  sphere_id = p.createMultiBody(baseVisualShapeIndex=target_visual_shape,
                                basePosition=np.array([0, 0, 0]))

  joint_ids = []
  param_ids = []

  p.setPhysicsEngineParameter(numSolverIterations=10)
  p.changeDynamics(reacher, -1, linearDamping=0, angularDamping=0)

  for j in range(p.getNumJoints(reacher)):
    p.changeDynamics(reacher, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(reacher, j)
    jointName = info[1]
    jointType = info[2]
    if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
      joint_ids.append(j)
      param_ids.append(
          p.addUserDebugParameter(jointName.decode("utf-8"), -math.pi, math.pi,
                                  0))

  if run_on_robot:
    serial_port = reacher_robot_utils.get_serial_port()
    hardware_interface = interface.Interface(serial_port)
    time.sleep(0.25)
    hardware_interface.set_joint_space_parameters(kp=KP,
                                                  kd=KD,
                                                  max_current=MAX_CURRENT)

  p.setRealTimeSimulation(1)
  counter = 0
  last_command = time.time()

  # Use this function to disable/enable certain motors. The first six elements
  # determine activation of the motors attached to the front of the PCB, which
  # are not used in this lab. The last six elements correspond to the activations
  # of the motors attached to the back of the PCB, which you are using.
  # The 7th element will correspond to the motor with ID=1, 8th element ID=2, etc
  # hardware_interface.send_dict({"activations": [0, 0, 0, 0, 0, 0, x, x, x, x, x, x]})

  while (1):
    if run_on_robot:
      hardware_interface.read_incoming_data()

    if time.time() - last_command > UPDATE_DT:
      print(time.time() - last_command)
      last_command = time.time()
      counter += 1
      joint_angles = np.zeros(3)
      for i in range(len(param_ids)):
        c = param_ids[i]
        targetPos = p.readUserDebugParameter(c)
        joint_angles[i] = targetPos
        p.setJointMotorControl2(reacher,
                                joint_ids[i],
                                p.POSITION_CONTROL,
                                targetPos,
                                force=2.)

      if run_on_robot:
        full_actions = np.zeros([3, 4])
        full_actions[:, 3] = np.reshape(joint_angles, 3)

        hardware_interface.set_actuator_postions(np.array(full_actions))
        # Actuator positions are stored in array: hardware_interface.robot_state.position,
        # Actuator velocities are stored in array: hardware_interface.robot_state.velocity

      end_effector_pos = reacher_kinematics.calculate_forward_kinematics_robot(
          joint_angles)
      p.resetBasePositionAndOrientation(sphere_id,
                                        posObj=end_effector_pos,
                                        ornObj=[0, 0, 0, 1])

      if counter % 5 == 0:
        print(
            reacher_kinematics.calculate_forward_kinematics_robot(joint_angles))


app.run(main)