import reacher_env
import math
import time
from absl import app
from absl import flags

flags.DEFINE_bool("run_on_robot", False,
                  "Whether to run on robot or in simulation.")
FLAGS = flags.FLAGS


def run_example():
  env = reacher_env.ReacherEnv(run_on_robot=FLAGS.run_on_robot)
  env.reset()
  env.setTarget([0.1, 0.1, 0.1])

  try:
    while True:
      #angles = env.calculateInverseKinematics(env.target)
      time.sleep(0.005)
      angles = [0, 0.2 * math.sin(time.time() * 2), 0]
      obs, reward, done, _ = env.step(actions=angles)
      print("obs: ", obs)
      #print("reward: ", reward)
  finally:
    env.shutdown()


def main(_):
  run_example()


if __name__ == "__main__":
  app.run(main)
