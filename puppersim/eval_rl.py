import argparse
import tqdm
import random
import pickle

import numpy as np
import gym
import torch
import gin

import super_sac
from super_sac_pupper import create_pupper_env, create_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=5_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", type=str, default=None, required=True)
    parser.add_argument("--num_rollouts", type=int, default=10)
    args = parser.parse_args()
    gin.parse_config_file(args.config)

    env = create_pupper_env(render=args.render, parallel_actors=1)
    agent = create_agent(env)
    agent.load(args.policy)
    agent.to(super_sac.device)

    super_sac.evaluation.evaluate_agent(
        agent,
        env,
        eval_episodes=args.num_rollouts,
        render=args.render,
        verbosity=1,
        sample_actions=False,
        max_episode_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
