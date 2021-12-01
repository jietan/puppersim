import pybullet as p
import time

import pybullet_data

def stateEstimator(pupper):
    xyz, quat_orien = p.getBasePositionAndOrientation(pupper)
    v_xyz, ang_vel = p.getBaseVelocity(pupper)

    # I need to transform these to pupper coordinates

    state_est = {"p":xyz, "p_d":v_xyz, "q":quat_orien, "w":ang_vel}



    return state_est