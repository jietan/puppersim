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
r"""An example that the Pupper stands.

"""
from absl import app
from absl import flags
import os
import time
import gin
from pybullet_envs.minitaur.agents.baseline_controller import static_gait_controller
from pybullet_envs.minitaur.envs_v2 import env_loader
import pybullet as p
import puppersim

import os


flags.DEFINE_bool("render", True, "Whether to render the example.")

FLAGS = flags.FLAGS
CONFIG_DIR = puppersim.getPupperSimPath()+"/"
_CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_with_imu.gin")
_NUM_STEPS = 10000
_ENV_RANDOM_SEED = 13


def _load_config(render=False):
  gin.parse_config_file(_CONFIG_FILE)
  gin.bind_parameter("SimulationParameters.enable_rendering", render)


def run_example(num_max_steps=_NUM_STEPS):
  """Runs the example.

  Args:
    num_max_steps: Maximum number of steps this example should run for.
  """
  env = env_loader.load()
  env.seed(_ENV_RANDOM_SEED)
  print("env.action_space=",env.action_space)
  observation = env.reset()
  policy = static_gait_controller.StaticGaitController(env.robot)
    
  for _ in range(num_max_steps):
    #action = policy.act(observation)
    action = [0, 0.6,-1.2,0, 0.6,-1.2,0, 0.6,-1.2,0, 0.6,-1.2]
    obs, reward, done, _ = env.step(action)
    time.sleep(0.01)
    if done:
      break


def main(_):
  _load_config(FLAGS.render)
  run_example()


if __name__ == "__main__":
  app.run(main)
