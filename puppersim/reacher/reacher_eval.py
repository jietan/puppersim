import argparse
import copy
import os
from typing import Tuple

import numpy as np
import gym

from reacher_env import ReacherEnv

from reacher_ddpg import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="reacher")
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    args = parser.parse_args()

    if args.env == "reacher":
        test_env = ReacherEnv(render=True)
    else:
        test_env = gym.make(args.env)

    test_env = gym.wrappers.TimeLimit(NormActionSpace(test_env), args.max_episode_steps)

    agent = Agent(
        test_env.observation_space.shape[0],
        test_env.action_space.shape[0],
        hidden_size=256,
    )
    agent.load(args.policy)
    agent.to(device)

    final_eval = evaluate_agent(
        agent,
        test_env,
        args.episodes,
        max_episode_steps=args.max_episode_steps,
        render=True,
    )
    print(f"Final Evaluation: {final_eval:.2f}")
