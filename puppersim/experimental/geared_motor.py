#a mimic joint can act as a gear between two joints
#you can control the gear ratio in magnitude and sign (>0 reverses direction)

import pybullet as p
import time
import pybullet_data
import numpy as np
p.connect(p.GUI)
dt = 1./240.

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw = 3.6,cameraPitch=-27, cameraTargetPosition=[0,0,0])
p.setPhysicsEngineParameter(numSolverIterations=200)
p.loadURDF("plane.urdf", 0, 0, -0.03)
motor = p.loadURDF("geared_motor.urdf", [0, 0, 0], flags=p.URDF_INITIALIZE_SAT_FEATURES)#, useFixedBase=True)
#p.changeDynamics(motor, -1, linearDamping=0, angularDamping=0, jointDamping=0)
for i in range(p.getNumJoints(motor)):
  print(p.getJointInfo(motor, i))
  p.changeDynamics(motor, i, linearDamping=0, angularDamping=0, jointDamping=0,maxJointVelocity=100000)
  p.setJointMotorControl2(motor, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

#p.setJointMotorControl2(motor, 0, p.VELOCITY_CONTROL, targetVelocity=1, force=10)

#p.resetJointState(motor,0,targetValue=0, targetVelocity=1)


class PDControllerExplicit(object):

  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep):
    numJoints = self._pb.getNumJoints(bodyUniqueId)
    jointStates = self._pb.getJointStates(bodyUniqueId, jointIndices)
    q1 = []
    qdot1 = []
    for i in range(len(jointStates)):
      q1.append(jointStates[i][0])
      qdot1.append(jointStates[i][1])
    q = np.array(q1)
    qdot = np.array(qdot1)
    qdes = np.array(desiredPositions)
    qdotdes = np.array(desiredVelocities)
    qError = (qdes - q)*0.
    qdotError = qdotdes - qdot
    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)
    forces = Kp.dot(qError) + Kd.dot(qdotError)
    maxF = np.array(maxForces)
    forces = np.clip(forces, -maxF, maxF)
    return forces

pd = PDControllerExplicit (p)

if 1:
  c = p.createConstraint(motor,
                           0,
                           motor,
                           1,
                           jointType=p.JOINT_GEAR,
                           jointAxis=[0, 0, 1],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, 0])
  p.changeConstraint(c, gearRatio=-1./36., maxForce=1000, erp=0.3)#,gearAuxLink=1)



while (1):
  p.setGravity(0, 0, -10)
  maxForces = [10]
  timeStep = dt
  kps = [0]
  kds = [0.1]
  desiredVelocities = [-10]
  desiredPositions = [0]
  linkIndices = [1]
  forces = pd.computePD(motor, linkIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep)
  p.setJointMotorControl2(motor, linkIndices[0], p.TORQUE_CONTROL,  force=forces[0])
  #print("forces=",forces)        
  jointIndices=[0,1]
  js = p.getJointStates(motor, jointIndices)
  #print("js=",js)
  p.stepSimulation()
  time.sleep(dt)

p.removeConstraint(c)
