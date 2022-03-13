import reacher_env
import math
import time

env = reacher_env.ReacherEnv()
env.reset()
env.setTarget([0.1, 0.1, 0.1])

while True:
    angles = env.calculateInverseKinematics(env.target)
    obs, reward, done, _ = env.step(actions=angles)
    # print("obs: ", obs)
    print("reward: ", reward)