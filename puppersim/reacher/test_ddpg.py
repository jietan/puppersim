import unittest

import gym
from reacher_env import ReacherEnv
from reacher_ddpg import *


class DDPG_General(unittest.TestCase):
    def setUp(self):
        self.train_env = gym.wrappers.TimeLimit(
            NormActionSpace(gym.make("Pendulum-v1")), 200
        )
        self.test_env = gym.wrappers.TimeLimit(
            NormActionSpace(gym.make("Pendulum-v1")), 200
        )

        self.agent = Agent(
            self.train_env.observation_space.shape[0],
            self.train_env.action_space.shape[0],
            hidden_size=128,
        )
        self.buffer = ReplayBuffer(
            10_000,
            example_state=self.train_env.reset(),
            example_action=self.train_env.action_space.sample(),
        )

    def test_ddpg_runs(self):
        ddpg(
            self.agent,
            self.train_env,
            self.test_env,
            self.buffer,
            num_steps=10,
            name="test_ddpg_runs",
            max_episode_steps=200,
            exploration_anneal=5,
            render=False,
            verbosity=0,
        )

    def test_ddpg_pendulum(self):
        _, best_return = ddpg(
            self.agent,
            self.train_env,
            self.test_env,
            self.buffer,
            num_steps=15_000,
            name="test_ddpg_pendlum",
            max_episode_steps=200,
            exploration_anneal=50_000,
            eval_interval=2000,
            return_best_score=True,
            render=False,
            verbosity=0,
        )
        self.assertTrue(best_return > -200.0)


class DDPG_Reacher(unittest.TestCase):
    def _setup(self, **env_kwargs):
        self.train_env = gym.wrappers.TimeLimit(
            NormActionSpace(ReacherEnv(**env_kwargs)), 500
        )
        self.test_env = gym.wrappers.TimeLimit(
            NormActionSpace(ReacherEnv(**env_kwargs)), 500
        )

        self.agent = Agent(
            self.train_env.observation_space.shape[0],
            self.train_env.action_space.shape[0],
            hidden_size=256,
        )
        self.buffer = ReplayBuffer(
            10_000,
            example_state=self.train_env.reset(),
            example_action=self.train_env.action_space.sample(),
        )

    def test_ddpg_reacher_position(self):
        self._setup(run_on_robot=False, render=False, torque_control=False)
        ddpg(
            self.agent,
            self.train_env,
            self.test_env,
            self.buffer,
            num_steps=10,
            name="test_ddpg_reacher_position",
            max_episode_steps=500,
            exploration_anneal=5,
            render=False,
            verbosity=0,
        )

    def test_ddpg_reacher_torque(self):
        self._setup(run_on_robot=False, render=False, torque_control=True)
        ddpg(
            self.agent,
            self.train_env,
            self.test_env,
            self.buffer,
            num_steps=10,
            name="test_ddpg_reacher_torque",
            max_episode_steps=500,
            exploration_anneal=5,
            render=False,
            verbosity=0,
        )
