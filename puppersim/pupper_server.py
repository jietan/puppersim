# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pybullet as p
import numpy as np
import time
import math

#if pip installed, we could use import puppersim.data as pd
import data as pd

import pybullet_data
from pybullet_utils import gazebo_world_parser
import pybullet_utils.urdfEditor as ed

cid = p.connect(p.GUI_SERVER)

#disable rendering during creation.
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

# disable keyboard shortcuts that allow you to toggle things like shadows (s) or wire-frame (w)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS,0) # s and w

# disable shadows to potentially increase speed
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

#white background
#p.configureDebugVisualizer(rgbBackground=[1,1,1])
p.setAdditionalSearchPath(pd.getDataPath())
p.setPhysicsEngineParameter(numSolverIterations=100)
p.setTimeStep(1. / 240.)

globalScaling=0.5
p.loadURDF(pybullet_data.getDataPath()+"/plane.urdf",[0,0,-0.01])
p.loadURDF("concrete_plane.urdf")

p.loadURDF("cone.urdf",[0.6,-0.1,0.0])
p.loadURDF("cone.urdf",[.98,0.,0.0])
p.loadURDF("cone.urdf",[1.34,-0.1,0.0])
p.loadURDF("cone.urdf",[1.62,0.,0.0])
p.loadURDF("cone.urdf",[1.9,-0.1,0.0])
ramp_orn1 = p.getQuaternionFromEuler([0,0,-math.pi/4])
ramp_orn2 = p.getQuaternionFromEuler([0,0,math.pi-math.pi/4])
p.loadURDF("ramp.urdf",[1-0.253,-1-0.253,0.11], ramp_orn1)
p.loadURDF("ramp.urdf",[1+0.253,-1+0.253,0.11], ramp_orn2)
load_parkour = False
if load_parkour:
  objects = [p.loadURDF("parkour/parkour.urdf", [globalScaling*2.000000,globalScaling*-2.000000,0.00000],[0.000000,0.000000,-0.707107,0.707107],useFixedBase=True, globalScaling=globalScaling)]
  ob = objects[0]
  jointPositions=[ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 ]
  for jointIndex in range (p.getNumJoints(ob)):
          p.resetJointState(ob,jointIndex,jointPositions[jointIndex])

# more complex environments can be loaded as well, check out
# see https://github.com/erwincoumans/aws-robomaker-racetrack-world
# gazebo_world_parser.parseWorld( p, filepath = "worlds/racetrack_day.world")

orn = p.getQuaternionFromEuler([0,0,-math.pi/2.])

#press 'G' or use the following line to hide the GUI
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

#disable tinyrenderer, software (CPU) renderer, we don't use it here
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

shift1 = [0, 0,0]
#shift2 = [0.088798, 0.160675, -0.010624 ]
shift2 = [0,0,0]
meshScale = [1, 1, 1]


robot = p.loadURDF("pupper_v2a.urdf",[0,0,0.23], p.getQuaternionFromEuler([0,0,math.pi/2.]), useFixedBase=False)

#p.changeVisualShape(robot,-1,rgbaColor=[1,0,0,1])
jointIds=[]
paramIds=[]          
                     
ABDUCTION_ANGLE=0
HIP_ANGLE=0.6
KNEE_ANGLE=-1.2
initial_joint_poses = [ABDUCTION_ANGLE,HIP_ANGLE,KNEE_ANGLE]*4
joint_index=0

jointNames=[]
for j in range (p.getNumJoints(robot)):
    p.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
    info = p.getJointInfo(robot, j)
    jointName = info[1]
    jointNames.append(jointName)
    print("jointName=", jointName)
    jointType = info[2]
    if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
        jointIds.append(j)
        paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, initial_joint_poses[joint_index]))
        joint_index+=1
                         
print("jointNames=",jointNames)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-134, cameraPitch=-30, cameraTargetPosition=[0.07,0.04,0.18])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


for i in range(len(paramIds)):
      c = paramIds[i]
      targetPos = p.readUserDebugParameter(c)
      p.setJointMotorControl2(robot, jointIds[i], p.VELOCITY_CONTROL, targetPos, force=0.02) # coulomb friction
      p.resetJointState(robot, jointIds[i], targetPos)


performance_benchmark = False
if performance_benchmark:
  #you can open the json file in Google Chrome or Microsoft Edge, 
  #visiting about://tracing and load the json
  logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "pupper_sim.json")
  for i in range (10):
    p.stepSimulation()
  p.stopStateLogging(logId)

#p.setRealTimeSimulation(1)
cam_skip=0
cam_skip_frames=1
p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
while p.isConnected():
    #if (cam_skip>cam_skip_frames):
    #  p.getCameraImage(320,200)#, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    #  cam_skip=0
    p.setGravity(0,0,-10)
    cam_skip+=1
    time.sleep(0.001)
