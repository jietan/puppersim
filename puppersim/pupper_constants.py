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

# Lint as: python3
"""Defines the Pupper robot related constants and URDF specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import numpy as np
import puppersim.data as pd
URDF_PATH = pd.getDataPath()+"/pupper_v2a.urdf" #or pupper_v2b with toes, but no visual meshes

NUM_MOTORS = 12
NUM_LEGS = 4
MOTORS_PER_LEG = 3

INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.17]

# Will be default to (0, 0, 0, 1) once the new laikago_toes_zup.urdf checked in.
INIT_ORIENTATION = [0, 0, 0, 1]

# Can be different from the motors
JOINT_NAMES = (
    # front right leg
    "rightFrontLegMotor",
    "rightFrontUpperLegMotor",
    "rightFrontLowerLegMotor",
    # front left leg
    "leftFrontLegMotor",
    "leftFrontUpperLegMotor",
    "leftFrontLowerLegMotor",
    # rear right leg
    "rightRearLegMotor",
    "rightRearUpperLegMotor",
    "rightRearLowerLegMotor",
    # rear left leg
    "leftRearLegMotor",
    "leftRearUpperLegMotor",
    "leftRearLowerLegMotor",
)

INIT_ABDUCTION_ANGLE = 0
INIT_HIP_ANGLE = 0.6
INIT_KNEE_ANGLE = -1.2

# Note this matches the Laikago SDK/control convention, but is different from
# URDF's internal joint angles which needs to be computed using the joint
# offsets and directions. The conversion formula is (sdk_joint_angle + offset) *
# joint direction.
INIT_JOINT_ANGLES = collections.OrderedDict(
    zip(JOINT_NAMES,
        (INIT_ABDUCTION_ANGLE, INIT_HIP_ANGLE, INIT_KNEE_ANGLE) * NUM_LEGS))

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_DIRECTIONS = collections.OrderedDict(
    zip(JOINT_NAMES, (-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1)))

HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0#-0.6
KNEE_JOINT_OFFSET = 0#0.66

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_OFFSETS = collections.OrderedDict(
    zip(JOINT_NAMES,
        [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] *
        NUM_LEGS))

LEG_NAMES = (
    "front_right",
    "front_left",
    "rear_right",
    "rear_left",
)

LEG_ORDER = (
    "front_right",
    "front_left",
    "back_right",
    "back_left",
)

END_EFFECTOR_NAMES = (
    "rightFrontToe",
    "leftFrontToe",
    "rightRearToe",
    "leftRearToe",
)

MOTOR_NAMES = JOINT_NAMES
MOTOR_GROUP = collections.OrderedDict((
    (LEG_NAMES[0], JOINT_NAMES[0:3]),
    (LEG_NAMES[1], JOINT_NAMES[3:6]),
    (LEG_NAMES[2], JOINT_NAMES[6:9]),
    (LEG_NAMES[3], JOINT_NAMES[9:12]),
))

# Regulates the joint angle change when in position control mode.
MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.02

# The hip joint location in the CoM frame.
HIP_POSITIONS = collections.OrderedDict((
    (LEG_NAMES[0], (0.21, -0.1157, 0)),
    (LEG_NAMES[1], (0.21, 0.1157, 0)),
    (LEG_NAMES[2], (-0.21, -0.1157, 0)),
    (LEG_NAMES[3], (-0.21, 0.1157, 0)),
))

MOTOR_ACTION_LOWER_LIMIT = np.array([-0.18,0.1,-2.3]*4)
MOTOR_ACTION_UPPER_LIMIT = np.array([0.18,0.7,-0.6]*4)
  
JOINT_FRICTION_FORCE = 0.021 # friction torque [Nm]

# Add the gin constants to be used for gin binding in config. Append "PUPPER_"
# for unique binding names.
gin.constant("pupper_constants.PUPPER_NUM_MOTORS", NUM_MOTORS)
gin.constant("pupper_constants.PUPPER_URDF_PATH", URDF_PATH)
gin.constant("pupper_constants.PUPPER_INIT_POSITION", INIT_POSITION)
gin.constant("pupper_constants.PUPPER_INIT_ORIENTATION", INIT_ORIENTATION)
gin.constant("pupper_constants.PUPPER_INIT_JOINT_ANGLES", INIT_JOINT_ANGLES)
gin.constant("pupper_constants.PUPPER_JOINT_DIRECTIONS", JOINT_DIRECTIONS)
gin.constant("pupper_constants.PUPPER_JOINT_OFFSETS", JOINT_OFFSETS)
gin.constant("pupper_constants.PUPPER_MOTOR_NAMES", MOTOR_NAMES)
gin.constant("pupper_constants.PUPPER_END_EFFECTOR_NAMES", END_EFFECTOR_NAMES)
gin.constant("pupper_constants.PUPPER_MOTOR_GROUP", MOTOR_GROUP)
gin.constant("pupper_constants.MOTOR_ACTION_LOWER_LIMIT", MOTOR_ACTION_LOWER_LIMIT)
gin.constant("pupper_constants.MOTOR_ACTION_UPPER_LIMIT", MOTOR_ACTION_UPPER_LIMIT)
gin.constant("pupper_constants.JOINT_FRICTION_FORCE", JOINT_FRICTION_FORCE)
