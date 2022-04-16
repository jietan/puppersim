import argparse
import copy
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import tqdm
import gym

from reacher_env import ReacherEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NormActionSpace(gym.ActionWrapper):
    """
    Standardize all envs to have actions in [-1, 1]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._true_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32,
        )

    def action(self, action):
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self.action_space.high - self.action_space.low
        action = (action - self.action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        return action


class Actor(nn.Module):
    """
    Policy Network (state --> action)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # actions bounded in (-1, 1)
        act = torch.tanh(self.out(x))
        return act


class Critic(nn.Module):
    """
    Value Network (state + action --> value)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.out(x)
        return val


class Agent:
    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 256,
    ):
        self.actor = Actor(state_size, action_size, hidden_size=hidden_size)
        self.critic = Critic(state_size, action_size, hidden_size=hidden_size)

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

    def save(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic_path = os.path.join(path, "critic.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, path):
        actor_path = os.path.join(path, "actor.pt")
        critic_path = os.path.join(path, "critic.pt")
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def forward(self, state):
        state = torch.from_numpy(state).to(device).unsqueeze(0).float()
        with torch.no_grad():
            action = self.actor(state)
        return action.squeeze(0).cpu().numpy()


def evaluate_agent(
    agent: Agent,
    env: gym.Env,
    episodes: int,
    max_episode_steps: int,
    render: bool = False,
):
    returns = []
    if render:
        env.render("rgb_array")
    for episode in range(episodes):
        episode_return = 0.0
        state = env.reset()
        done, info = False, {}
        for step_num in range(max_episode_steps):
            if done:
                break
            action = agent.forward(state)
            state, reward, done, info = env.step(action)
            if render:
                env.render("rgb_array")
            episode_return += reward
        returns.append(episode_return)
    mean_return = np.array(returns).mean().item()
    return mean_return


class ReplayBuffer:
    """
    Store environment experience to train the agent.
    """

    def __init__(
        self, size: int, example_state: np.ndarray, example_action: np.ndarray
    ):
        self.s_stack = np.zeros(
            (size,) + example_state.shape, dtype=example_state.dtype
        )
        self.a_stack = np.zeros(
            (size,) + example_action.shape, dtype=example_action.dtype
        )
        self.r_stack = np.zeros((size, 1), dtype=np.float32)
        self.s1_stack = np.zeros(
            (size,) + example_state.shape, dtype=example_state.dtype
        )
        self.d_stack = np.zeros((size, 1), dtype=bool)

        self.size = size
        self._next_idx = 0
        self._max_filled = 0

    def __len__(self):
        return self._max_filled

    def push(self, s, a, r, s1, d):
        idx = self._next_idx
        self.s_stack[idx] = s
        self.a_stack[idx] = a
        self.r_stack[idx] = np.array([r])
        self.s1_stack[idx] = s1
        self.d_stack[idx] = np.array([d])
        self._max_filled = min(max(self._next_idx + 1, self._max_filled), self.size)
        self._next_idx = (self._next_idx + 1) % self.size

    def torch_and_move(self, *np_ndarrays):
        return (torch.from_numpy(x).to(device).float() for x in np_ndarrays)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor]:
        idxs = np.random.randint(0, len(self), size=(batch_size,))
        state = self.s_stack[idxs]
        action = self.a_stack[idxs]
        reward = self.r_stack[idxs]
        next_state = self.s1_stack[idxs]
        done = self.d_stack[idxs]
        return self.torch_and_move(state, action, reward, next_state, done)


class ExplorationNoise:
    """
    We encourage exploration by adding random noise to our actions during training.
    """

    def __init__(self, size, start_scale=1.0, final_scale=0.1, steps_annealed=1000):
        assert start_scale >= final_scale
        self.size = size
        self.start_scale = start_scale
        self.final_scale = final_scale
        self.steps_annealed = steps_annealed
        self._current_scale = start_scale
        self._scale_slope = (start_scale - final_scale) / steps_annealed

    def sample(self):
        noise = self._current_scale * np.random.randn(*self.size)
        self._current_scale = max(
            self._current_scale - self._scale_slope, self.final_scale
        )
        return noise


def make_save_folder(run_name, base_path="rl_agents"):
    base_dir = os.path.join(base_path, run_name)
    i = 0
    while os.path.exists(base_dir + f"_{i}"):
        i += 1
    base_dir += f"_{i}"
    os.makedirs(base_dir)
    return base_dir


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    Used to make the `target` NN's params a moving average of the `source` NN's params.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    """
    Hard copy `source` params to `target`.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def ddpg(
    agent: Agent,
    train_env: gym.Env,
    test_env: gym.Env,
    buffer: ReplayBuffer,
    num_steps: int = 1_000_000,
    transitions_per_step: int = 5,
    max_episode_steps: int = 100_000,
    batch_size: int = 512,
    tau: float = 0.005,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-4,
    gamma: float = 0.99,
    exploration_anneal: int = 100_000,
    eval_interval: int = 5000,
    eval_episodes: int = 20,
    warmup_steps: int = 1000,
    name: str = "ddpg_run",
    gradient_updates_per_step: int = 1,
    infinite_bootstrap: bool = True,
    return_best_score: bool = False,
    render: bool = True,
    verbosity: int = 1,
) -> Agent:
    """
    Train `agent` on `train_env` with the Deep Deterministic Policy Gradient algorithm,
    and evaluate on `test_env`. Returns the best performing agent.

    Reference: https://arxiv.org/abs/1509.02971
    """
    best_return = -float("inf")
    save_dir = make_save_folder(name)

    agent.to(device)

    # initialize target networks
    target_agent = copy.deepcopy(agent)
    target_agent.to(device)
    hard_update(target_agent.actor, agent.actor)
    hard_update(target_agent.critic, agent.critic)

    random_process = ExplorationNoise(
        size=train_env.action_space.shape, steps_annealed=exploration_anneal
    )

    critic_optimizer = torch.optim.Adam(agent.critic.parameters(), lr=critic_lr)
    actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=actor_lr)

    done = True

    train_iter = range(num_steps)
    if verbosity:
        train_iter = tqdm.tqdm(train_iter)
    for step in train_iter:
        for _ in range(transitions_per_step):
            # collect experience from the environment, sampling from
            # the current policy (with added noise for exploration)
            if done:
                # reset the environment
                state = train_env.reset()
                steps_this_ep = 0
                done = False
            action = agent.forward(state)
            noisy_action = np.clip(action + random_process.sample(), -1.0, 1.0)
            next_state, reward, done, info = train_env.step(noisy_action)
            if infinite_bootstrap:
                # allow infinite bootstrapping. Many envs terminate
                # (done = True) after an arbitrary number of steps
                # to let the agent reset and avoid getting stuck in
                # a failed position. infinite bootstrapping prevents
                # this from impacting our Q function calculation. This
                # can be harmful in edge cases where the environment really
                # would have ended (task failed) regardless of the step limit,
                # and makes no difference if the environment is not set up
                # to enforce a limit by itself (but many common benchmarks are).
                if steps_this_ep + 1 == max_episode_steps:
                    done = False
            # add this transition to the replay buffer
            buffer.push(state, noisy_action, reward, next_state, done)
            state = next_state
            steps_this_ep += 1
            if steps_this_ep >= max_episode_steps:
                # enforce max step limit from the agent's perspective
                done = True

        if len(buffer) > warmup_steps:
            for _ in range(gradient_updates_per_step):
                # update the actor and critics using the replay buffer
                learn(
                    buffer=buffer,
                    target_agent=target_agent,
                    agent=agent,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    batch_size=batch_size,
                    gamma=gamma,
                )

                # move target models towards the online models
                soft_update(target_agent.actor, agent.actor, tau)
                soft_update(target_agent.critic, agent.critic, tau)

        if step % eval_interval == 0 or step == num_steps - 1:
            mean_return = evaluate_agent(
                agent, test_env, eval_episodes, max_episode_steps, render=render,
            )
            if verbosity:
                print(f"Mean Return After {step} Steps: {mean_return:.2f}")
            if mean_return > best_return:
                agent.save(save_dir)
                best_return = mean_return

    # restore best saved agent
    agent.load(save_dir)
    if return_best_score:
        return agent, best_return
    return agent


def learn(
    buffer, target_agent, agent, actor_optimizer, critic_optimizer, batch_size, gamma,
):
    """
    DDPG inner optimization loop. The simplest deep
    actor critic update.
    """
    (
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
    ) = buffer.sample(batch_size)

    ###################
    ## Critic Update ##
    ###################

    # compute target values
    with torch.no_grad():
        target_action_s1 = target_agent.actor(next_state_batch)
        target_action_value_s1 = target_agent.critic(next_state_batch, target_action_s1)
        # bootstrapped estimate of Q(s, a) based on reward and target network
        td_target = reward_batch + gamma * (1.0 - done_batch) * target_action_value_s1

    # compute mean squared bellman error (MSE(Q(s, a), td_target))
    agent_critic_pred = agent.critic(state_batch, action_batch)
    td_error = td_target - agent_critic_pred
    critic_loss = 0.5 * (td_error ** 2).mean()
    critic_optimizer.zero_grad()
    # gradient descent step on critic network
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 10.0)
    critic_optimizer.step()

    ##################
    ## Actor Update ##
    ##################

    # actor's objective is to maximize (or minimize the negative of)
    # the expectation of the critic's opinion of its action choices
    agent_actions = agent.actor(state_batch)
    actor_loss = -agent.critic(state_batch, agent_actions).mean()
    actor_optimizer.zero_grad()
    # gradient descent step on actor network
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 10.0)
    actor_optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="reacher")
    parser.add_argument("--train_steps", type=int, default=150_000)
    parser.add_argument("--name", type=str, default="ddpg_run")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    args = parser.parse_args()

    if args.env == "reacher":
        train_env = ReacherEnv(render=False)
        test_env = ReacherEnv(render=False)
    else:
        train_env = gym.make(args.env)
        test_env = gym.make(args.env)

    train_env = gym.wrappers.TimeLimit(
        NormActionSpace(train_env), args.max_episode_steps
    )
    test_env = gym.wrappers.TimeLimit(NormActionSpace(test_env), args.max_episode_steps)

    agent = Agent(
        train_env.observation_space.shape[0],
        train_env.action_space.shape[0],
        hidden_size=256,
    )

    buffer = ReplayBuffer(
        size=500_000,
        example_state=train_env.reset(),
        example_action=train_env.action_space.sample(),
    )

    agent = ddpg(
        agent,
        train_env,
        test_env,
        buffer,
        num_steps=args.train_steps,
        name=args.name,
        max_episode_steps=args.max_episode_steps,
        exploration_anneal=args.train_steps // 2,
        render=False,
    )

    final_eval = evaluate_agent(
        agent, test_env, 10, max_episode_steps=args.max_episode_steps, render=True,
    )
    print(f"Final Evaluation: {final_eval:.2f}")
