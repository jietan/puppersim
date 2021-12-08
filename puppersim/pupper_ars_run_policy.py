"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
python3 pupper_ars_run_policy.py --expert_policy_file=data/lin_policy_plus_best_10.npz --json_file=data/params.json

"""
import numpy as np
import gym
import time
import pybullet_envs
try:
  import tds_environments
except:
  pass
import json
from arspb.policies import *
import time
import arspb.trained_policies as tp
import os

#temp hack to create an envs_v2 pupper env

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
from pybullet import COV_ENABLE_GUI
import puppersim.data as pd

def create_pupper_env(args):
  CONFIG_DIR = puppersim.getPupperSimPath()
  if args.run_on_robot:
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg_robot.gin")
  else:
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg.gin")
  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(_CONFIG_FILE)
  gin.bind_parameter("SimulationParameters.enable_rendering", args.render)
  env = env_loader.load()
  env._pybullet_client.configureDebugVisualizer(COV_ENABLE_GUI, 0)
  
  return env


def main(argv):
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--expert_policy_file', type=str, default="data/lin_policy_plus_latest.npz", help='relative path to the policy weights. Defaults to where ars_train outputs weight file.')

  parser.add_argument('--num_rollouts', type=int, default=20,
                      help='Number of expert rollouts. Default is 20.')
  parser.add_argument('--json_file', type=str, default="data/params.json", help='relative path to the policy parameters file. Defaults to where ars_train outputs params file.')
  parser.add_argument('--run_on_robot', action='store_true', help='whether to run the policy on the robot instead of in simulation. Default is False.')
  parser.add_argument('--render', default=False, action='store_true', help='whether to render the robot. Default is False.')
  parser.add_argument('--profile', default=False, action='store_true', help='whether to print timing for parts of the code. Default is False.')
  parser.add_argument('--plot', default=True, action='store_true', help='whether to plot action and observation histories after running the policy.')
  if len(argv):
    args = parser.parse_args(argv)
  else:
    args = parser.parse_args()

  print('loading and building expert policy')
  if len(args.json_file)==0:
    args.json_file = tp.getDataPath()+"/"+ args.envname+"/params.json"    
  with open(args.json_file) as f:
      params = json.load(f)
  print("params=",params)
  if len(args.expert_policy_file)==0:
    args.expert_policy_file=tp.getDataPath()+"/"+args.envname+"/nn_policy_plus.npz" 
    if not os.path.exists(args.expert_policy_file):
      args.expert_policy_file=tp.getDataPath()+"/"+args.envname+"/lin_policy_plus.npz"
  data = np.load(args.expert_policy_file, allow_pickle=True)

  print('create gym environment:', params["env_name"])
  env = create_pupper_env(args)#gym.make(params["env_name"])


  lst = data.files
  weights = data[lst[0]][0]
  mu = data[lst[0]][1]
  print("mu=",mu)
  std = data[lst[0]][2]
  print("std=",std)

  ob_dim = env.observation_space.shape[0]
  ac_dim = env.action_space.shape[0]
  ac_lb = env.action_space.low
  ac_ub = env.action_space.high

  policy_params={'type': params["policy_type"],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim,
                   'action_lower_bound' : ac_lb,
                   'action_upper_bound' : ac_ub,
  }
  policy_params['weights'] = weights
  policy_params['observation_filter_mean'] = mu
  policy_params['observation_filter_std'] = std
  if params["policy_type"]=="nn":
    print("FullyConnectedNeuralNetworkPolicy")
    policy_sizes_string = params['policy_network_size_list'].split(',')
    print("policy_sizes_string=",policy_sizes_string)
    policy_sizes_list = [int(item) for item in policy_sizes_string]
    print("policy_sizes_list=",policy_sizes_list)
    policy_params['policy_network_size'] = policy_sizes_list
    policy = FullyConnectedNeuralNetworkPolicy(policy_params, update_filter=False)
  else:
    print("LinearPolicy2")
    policy = LinearPolicy2(policy_params, update_filter=False)
  policy.get_weights()

  returns = []
  observations = []
  actions = []
  for i in range(args.num_rollouts):
    print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    start_time_wall = time.time()
    while not done or args.run_on_robot:
      if time.time() - start_time_wall > env.env_time_step:
        if args.profile:
          print("loop dt:", time.time() - start_time_wall)
        start_time_wall = time.time()
        before_policy = time.time()
        action = policy.act(obs)
        after_policy = time.time()

        if not args.run_on_robot:
          observations.append(obs)
          actions.append(action)

        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if args.profile:
          print('policy.act(obs): ', after_policy - before_policy)
          print('wallclock_control_code: ', time.time() - start_time_wall)
    returns.append(totalr)

  print('returns: ', returns)
  print('mean return: ', np.mean(returns))
  print('std of return: ', np.std(returns))

  if args.plot and not args.run_on_robot:
    import matplotlib.pyplot as plt
    action_history = np.array(actions)
    observation_history = np.array(observations)
    plt.plot(action_history)
    plt.show()

if __name__ == '__main__':
  import sys
  main(sys.argv[1:])
