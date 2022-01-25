import argparse
import os
import gym
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd
import cv2

from super_sac.nets import distributions, weight_init

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym

import puppersim
import pybullet
import pybullet_envs

import super_sac
from super_sac.wrappers import (
    SimpleGymWrapper,
    NormActionSpace,
    ParallelActors,
    ScaleReward,
    FrameStack,
    FrameSkip,
    StateStack,
    Uint8Wrapper,
)


class AddActionReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.concatenate(
            (env.observation_space.low, env.action_space.low, np.array([-np.inf]))
        )
        high = np.concatenate(
            (env.observation_space.low, env.action_space.high, np.array([np.inf]))
        )
        shape = (env.observation_space.shape[0] + env.action_space.shape[0] + 1,)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=env.observation_space.dtype,
        )

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        obs = np.concatenate((obs, act, np.array([rew])), axis=0)
        return obs, rew, done, info

    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs, np.zeros(self.action_space.shape), np.array([0.0])))


class AddPosition(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.concatenate(
            (env.observation_space.low, np.array([-np.inf, np.inf, -np.inf]))
        )
        high = np.concatenate(
            (env.observation_space.high, np.array([np.inf, np.inf, np.inf]))
        )
        shape = (env.observation_space.shape[0] + 3,)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        state = self.env.reset()
        return self.observation(state)

    def step(self, action):
        state, rew, done, info = self.env.step(action)
        state = self.observation(state)
        return state, rew, done, info

    def observation(self, obs):
        position = np.array(self.env._last_base_position, dtype=np.float32) / 10.0
        return np.concatenate((obs, position))


class PupperFromVision(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._img = None
        self.env = env
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 84, 84))
        self.reset()

    def reset(self):
        self.env.reset()
        self._img = self.env.render("rgb_array").transpose(2, 0, 1)
        return self._img

    RENDER_DELAY = 1

    def render(self, *args, **kwargs):
        cv2.imshow("Puppersim", self._img.transpose(1, 2, 0))
        cv2.waitKey(self.RENDER_DELAY)

    def step(self, action):
        next_state, rew, done, info = self.env.step(action)
        self._img = self.env.render("rgb_array").transpose(2, 0, 1)
        return self._img, rew, done, info


class NoRender(gym.Wrapper):
    def render(self, *args, **kwargs):
        pass


@gin.configurable
def create_pupper_env(render=False, from_pixels=False, skip=1, stack=1):
    # build env from pybullet config
    CONFIG_DIR = puppersim.getPupperSimPath() + "/config/"
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_manual_rl.gin")
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath() + "/")
    gin.parse_config_file(_CONFIG_FILE)
    if render:
        gin.bind_parameter("SimulationParameters.enable_rendering", True)
    env = env_loader.load()

    if from_pixels:
        env = PupperFromVision(env)
        env = FrameSkip(env, skip=skip)
        env = FrameStack(env, num_stack=stack)
        env = NormActionSpace(env)
    else:
        env = AddPosition(env)
        env = NormActionSpace(env)
        env = AddActionReward(env)
        env = StateStack(env, num_stack=stack, skip=skip)
        env = NoRender(env)

    env = ScaleReward(env, scale=100.0)

    if from_pixels:
        env = Uint8Wrapper(env)
    else:
        env = SimpleGymWrapper(env)
    return env


@gin.configurable
class CCEncoder(super_sac.nets.Encoder):
    def __init__(self, obs_space):
        super().__init__()
        self._dim = obs_space.shape[0]

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs_dict):
        return obs_dict["obs"]


@gin.configurable
class LSTMEncoder(super_sac.nets.Encoder):
    def __init__(self, obs_space, out_dim=50, seq_length=20, hidden_size=256):
        super().__init__()
        inp_dim = obs_space.shape[0] // seq_length
        self.fc_in = nn.Linear(inp_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.fc = nn.Linear(hidden_size, out_dim)
        self.seq_length = seq_length
        self._dim = out_dim

    @property
    def embedding_dim(self):
        return self._dim

    def forward(self, obs):
        seq = obs["obs"]
        seq = torch.cat(seq.unsqueeze(1).chunk(self.seq_length, dim=-1)[::-1], dim=1)
        emb_seq = torch.tanh(self.ln1(self.fc_in(seq)))
        out, (h, c) = self.lstm(emb_seq)
        emb = F.relu(self.fc(h.squeeze(0)))
        return emb


@gin.configurable
class VisionEncoder(super_sac.nets.Encoder):
    def __init__(self, obs_space, emb_dim=50):
        super().__init__()
        self._dim = emb_dim
        self.conv_block = super_sac.nets.cnns.BigPixelEncoder(obs_space.shape, emb_dim)

    def forward(self, obs_dict):
        emb = self.conv_block(obs_dict["obs"])
        return emb

    @property
    def embedding_dim(self):
        return self._dim


@gin.configurable
def create_agent(train_env, encoder_cls=None):
    assert encoder_cls is not None
    encoder = encoder_cls(train_env.observation_space)
    agent = super_sac.Agent(
        act_space_size=train_env.action_space.shape[0],
        encoder=encoder,
    )
    return agent


def train_pupper(name, logging_method):
    train_env = create_pupper_env()
    test_env = create_pupper_env()
    agent = create_agent(train_env)
    buffer = super_sac.replay.ReplayBuffer(size=1_000_000)
    super_sac.super_sac(
        agent=agent,
        train_env=train_env,
        test_env=test_env,
        buffer=buffer,
        name=name,
        logging_method=logging_method,
    )
    return agent


def main(args):
    gin.parse_config_file(args.config)
    train_pupper(args.name, args.logging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--logging", type=str, default="wandb", choices=["wandb", "tensorboard"]
    )
    args = parser.parse_args()
    main(args)
