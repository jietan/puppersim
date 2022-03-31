import reacher_env
import math
import time
import numpy as np
from absl import app
from absl import flags
import copy


flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
FLAGS = flags.FLAGS

HIP_OFFSET = 0.0335
L1 = 0.08
L2 = 0.11

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

def ik_cost(end_effector_pos, guess):
    return 0.5 * np.linalg.norm(calculate_forward_kinematics_robot(guess)-end_effector_pos) ** 2

def calculate_inverse_kinematics(end_effector_pos, guess):
    # compute joint angles given a desired end effector position
    lmbda = 10
    cost = ik_cost(end_effector_pos, guess)
    iters = 0
    while cost > 1e-6 and iters < 100:
        iters += 1
        J = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                h = 1e-5
                p2 = copy.deepcopy(guess)
                p2[j] += h
                J[i, j] = (calculate_forward_kinematics_robot(p2)[i] - calculate_forward_kinematics_robot(guess)[i]) / h
        dif = np.reshape(calculate_forward_kinematics_robot(guess) - end_effector_pos, (3,1))
        J_t = np.transpose(J)
        cost_gradient = np.matmul(J_t, dif)
        guess = np.reshape(guess, (3,1)) - lmbda * cost_gradient
        cost = ik_cost(end_effector_pos, guess)
        if iters % 100 == 0:
            print("iters", iters, "cost: ", cost)
    print(calculate_forward_kinematics_robot(guess), end_effector_pos)
    return guess


    

def run_example():
    env = reacher_env.ReacherEnv(run_on_robot=FLAGS.run_on_robot, render=True)
    env.reset()
    env.setTarget([0.0, 0.0, 0.15])
    guess = [0,0,0]

    env_step = 0
    cumulative_reward = 0
    while True:
        env_step += 1
        time.sleep(0.002)
        desired_end_effector_pos = env.target
        print("g", guess)
        guess = calculate_inverse_kinematics(desired_end_effector_pos, guess)
        obs, reward, done, _ = env.step(actions=guess)
        cumulative_reward += reward
        print("obs: ", obs)
        print("reward: ", cumulative_reward)
        if env_step > 1000:
            cumulative_reward = 0
            env_step = 0
            env.reset()
    
    
def main(_):
  run_example()


if __name__ == "__main__":
  app.run(main)