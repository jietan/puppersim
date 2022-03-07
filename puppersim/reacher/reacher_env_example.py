import reacher_env
import math
import time

env = reacher_env.ReacherEnv()
env.reset()

# TODO write IK to find desired action
# TODO set action so that end effector reaches target (reward should be 0)

while True:
    obs, reward, done, _ = env.step(actions=[0, 0, math.sin(time.time())])
    print("obs: ", obs)
    print("reward: ", reward)