import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import puppersim
import os
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

def create_pupper_env():
  CONFIG_DIR = puppersim.getPupperSimPath()
  _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
  #  _NUM_STEPS = 10000
  #  _ENV_RANDOM_SEED = 2

  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(_CONFIG_FILE)
  env = env_loader.load()
  return env


class PupperGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self):
    self.env = create_pupper_env()
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space

  #def _configure(self, display=None):
  #  self.display = display

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    s = int(seed) & 0xffffffff
    self.env.seed(s)

    return [seed]

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    return self.env.reset()

  def update_weights(self, weights):
    self.env.update_weights(weights)

  def render(self, mode='human', close=False,  **kwargs):
    return self.env.render(mode)

  def configure(self, args):
    self.env.configure(args)

  def close(self):
    self.env.close()

    