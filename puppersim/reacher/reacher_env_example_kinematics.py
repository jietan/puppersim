import reacher_env
from puppersim.reacher import reacher_kinematics
import math
import time
import numpy as np
from absl import app
from absl import flags
import copy

flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
FLAGS = flags.FLAGS    

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
        guess = reacher_kinematics.calculate_inverse_kinematics(desired_end_effector_pos, guess)
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