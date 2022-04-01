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
    env_step = 0
    while True:
      env_step += 1
      time.sleep(0.005)
      angles = [0, 0, math.sin(2*time.time())]
      obs, reward, done, _ = env.step(actions=angles)
      if env_step > 1000:
        env.reset()
        env_step = 0
      print("obs: ", obs)
      print("reward: ", reward)
  finally:
    env.shutdown()


def main(_):
  run_example()


if __name__ == "__main__":
  app.run(main)
