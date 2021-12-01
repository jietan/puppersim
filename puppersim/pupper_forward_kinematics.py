    
import numpy as np
import math

def getPos(joint_angles):
    l1 = 0.08
    l2 = 0.12
    hip_offset = 0.035

    print(joint_angles[1])
    p1x = -l1 * math.sin(joint_angles[1])
    p1z = -l1 * math.cos(joint_angles[1])

    p2x = -l2 * math.sin(joint_angles[1] + joint_angles[2])
    p2z = -l2 * math.cos(joint_angles[1] + joint_angles[2])

    foot_pos = np.array([p1x + p2x, hip_offset, p1z + p2z])

    rot_mat = np.array([[1, 0, 0],
        [0, math.cos(joint_angles[0]), -math.sin(joint_angles[0])], 
        [0, math.sin(joint_angles[0]), math.cos(joint_angles[0])]])

    xyz = np.matmul(rot_mat, foot_pos)

    return xyz