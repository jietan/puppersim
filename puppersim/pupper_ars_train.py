import ray
from packaging import version
import socket

from arspb import ars

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', type=str, default='PupperGymEnv-v0')
  parser.add_argument('--n_iter', '-n', type=int, default=1000)
  parser.add_argument('--n_directions', '-nd', type=int, default=16)
  parser.add_argument('--deltas_used', '-du', type=int, default=16)
  parser.add_argument('--step_size', '-s', type=float, default=0.03)
  parser.add_argument('--delta_std', '-std', type=float, default=.03)
  parser.add_argument('--n_workers', '-e', type=int, default=18)
  parser.add_argument('--rollout_length', '-r', type=int, default=200)
  parser.add_argument('--shift', type=float, default=0)
  parser.add_argument('--seed', type=int, default=37)
  parser.add_argument('--policy_type',
                      type=str,
                      help='Policy type, linear or nn (neural network)',
                      default='linear')
  parser.add_argument('--dir_path', type=str, default='data')

  # for ARS V1 use filter = 'NoFilter'
  parser.add_argument('--filter', type=str, default='MeanStdFilter')
  parser.add_argument(
      '--activation',
      type=str,
      help='Neural network policy activation function, tanh or clip',
      default='tanh')
  parser.add_argument('--policy_network_size',
                      action='store',
                      dest='policy_network_size_list',
                      type=str,
                      default='64,64')
  parser.add_argument('--redis_address',
                      type=str,
                      default=socket.gethostbyname(socket.gethostname()) +
                      ':6379')

  args = parser.parse_args()

  print('redis_address=', args.redis_address)
  if version.parse(ray.__version__) >= version.parse("1.6.0"):
    ray.init(address=args.redis_address)
  else:
    ray.init(redis_address=args.redis_address)

  params = vars(args)
  ars.run_ars(params)
