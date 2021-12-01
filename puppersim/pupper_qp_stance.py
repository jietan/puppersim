import pybullet as p
import time

import pybullet_data

from pupper_forward_kinematics import getPos
from pupper_state_estimator import stateEstimator

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
pupper = p.loadURDF("data/pupper_v2a.urdf", useFixedBase=False)
p.loadURDF(pybullet_data.getDataPath()+"/plane.urdf",[0,0,-0.01])


gravId = p.addUserDebugParameter("gravity", -10, 10, -10)
jointIds = []
paramIds = []

p.setPhysicsEngineParameter(numSolverIterations=10)
p.changeDynamics(pupper, -1, linearDamping=0, angularDamping=0)

for j in range(p.getNumJoints(pupper)):
  p.changeDynamics(pupper, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(pupper, j)
  #print(info)
  jointName = info[1]
  jointType = info[2]
  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    jointIds.append(j)
    paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

def getLegsPos(pupper, jointIds):
  positions = []
  for leg in range(4):
    joints = []
    for i in range(3):
      joints.append(p.getJointState(pupper,jointIds[leg*3+i])[0])
    positions.append(joints)
  return positions

xyz_desired = stateEstimator(pupper)["p"] # set to initial, then add to sin(t)
mass = 2
inertia = 3

j=1
while (1):
  j+=1
  p.setGravity(0, 0, p.readUserDebugParameter(gravId))

  # set desired angular/ cartesian accelerations

  state = stateEstimator(pupper)
  xyz = state["p"]
  v_xyz = state["p_d"]
  quat_orien = state["q"]
  ang = p.getEulerFromQuaternion(quat_orien)
  ang_vel = state["w"]

  kp_xyz = 1.0
  xyz_v_desired = (0, 0, 0)

  a = tuple(map(lambda i, j: kp_xyz * (i - j), xyz, xyz_desired))

  f_xyz = mass * a

  kp_ang = 400
  ang_desired = (0, 0, 0)

  a_ang = tuple(map(lambda i, j: kp_ang * (i - j), ang, ang_desired))
  tau = inertia * a_ang # figure out inertia of pupper?

  if j%60==0:
    print(ang)

  for i in range(len(paramIds)):
    c = paramIds[i]
    targetPos = p.readUserDebugParameter(c)
    p.setJointMotorControl2(pupper, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)

  time.sleep(0.005)
  p.stepSimulation()

# p.setRealTimeSimulation(1)
j=1
# while (1):
#   p.setGravity(0, 0, p.readUserDebugParameter(gravId))
#   leg1_joints = [p.getJointState(pupper,jointIds[0])[0], p.getJointState(pupper,jointIds[1])[0], p.getJointState(pupper,jointIds[2])[0]]
#   LFPos, RFPos, LBPos, RFPos = getLegsPos(pupper, jointIds)
#   if j%100==0:
#     leg1_joints = [p.getJointState(pupper,jointIds[0])[0], p.getJointState(pupper,jointIds[1])[0], p.getJointState(pupper,jointIds[2])[0]]
#     xyz=getPos(leg1_joints)
#     print(xyz)
#   for i in range(len(paramIds)):
#     c = paramIds[i]
#     # torque = 100*p.readUserDebugParameter(c)
#     targetPos = p.readUserDebugParameter(c)
#     p.setJointMotorControl2(pupper, jointIds[i], p.POSITION_CONTROL, targetPos, force=5 * 240.)
#     # p.setJointMotorControl2(pupper, jointIds[i], p.TORQUE_CONTROL, force=torque)
#   time.sleep(0.005)
#   p.stepSimulation()
j+=1
