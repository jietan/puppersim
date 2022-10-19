from collections.abc import Sequence

import actor_critic
import torch

from absl import app
from absl import flags
import os
import time
import gin
import math
import numpy as np
import pickle
from pybullet_envs.minitaur.envs_v2 import env_loader
import pybullet as p
import puppersim

import os


flags.DEFINE_bool("render", True, "Whether to render the example.")
flags.DEFINE_bool("profile", False, "Whether to print timing results for different parts of the code.")
flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
flags.DEFINE_bool("log_to_file", False, "Whether to log data to the disk.")
flags.DEFINE_bool("realtime", False, "Run at realtime.")
FLAGS = flags.FLAGS
CONFIG_DIR = puppersim.getPupperSimPath()
_NUM_STEPS = 100000
_ENV_RANDOM_SEED = 13

MAX_TORQUE = 1.7 # Maximum torque in [Nm]. For DJI Pupper limited to 7A, the maximum torque pushing against resistance is ~1.7Nm.

def _load_config(render=False):
  if FLAGS.run_on_robot:
    config_file = os.path.join(CONFIG_DIR, "config", "pupper_robot.gin")
  else:
    config_file = os.path.join(CONFIG_DIR, "config", "pupper.gin")

  gin.parse_config_file(config_file)
  gin.bind_parameter("SimulationParameters.enable_rendering", render)


class IsaacGymPolicy(object):

  def __init__(self, checkpoint_path, device):

    num_obs = 31  # Need to find the list of sensors being used for pupper.
    num_critic_obs = 31  # If no priviledge info is specified
    num_actions = 12

    # To match the training config.
    policy_cfg = {
        'activation': 'elu',
        'actor_hidden_dims': [512, 256, 128],
        'critic_hidden_dims': [512, 256, 128],
        'init_noise_std': 1.0
    }
    actor_critic_policy = actor_critic.ActorCritic(num_obs, num_critic_obs,
                                                   num_actions,
                                                   **policy_cfg).to(device)

    loaded_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    actor_critic_policy.load_state_dict(loaded_dict['model_state_dict'])

    actor_critic_policy.eval()
    self.last_action = np.zeros(12)
    self.default_dof_pos = np.array([0, 0.5, -1.2] * 4)

    self.device = device
    self.policy = actor_critic_policy.act_inference
  
  def convert_obs_for_policy(self, obs):
    orientation_scale = 1 
    angular_velocity_scale = 0.25
    joint_angle_scale = 1 
     
    command_scale = np.array([2, 2, 0.25])
    last_action_scale = 1
   
    rp = torch.tensor(obs['IMU'][:2] * orientation_scale)
    rp_dot = torch.tensor(obs['IMU'][2:] * angular_velocity_scale)
    motor_angle = torch.tensor((obs['MotorAngle'] - self.default_dof_pos) * joint_angle_scale) 
    command = torch.tensor(obs['command'] * command_scale)
    last_action = torch.tensor(self.last_action * last_action_scale)
    new_obs =  torch.cat([rp, rp_dot, motor_angle, command, last_action])
    
    return torch.tensor(new_obs, device=self.device)

  def step(self, obs):
    # The following information will be pulled from the pupper training config
    clip_actions = 100
    action_scale = 0.3
    default_dof_pos = torch.tensor(self.default_dof_pos, device=self.device)

    new_obs = self.convert_obs_for_policy(obs)
    # print('new_obs', new_obs)
    # Double check if the action is batched or not.
    action = self.policy(new_obs.detach())
    # print('original_action', action)
    self.last_action = action
    

    # Now needs to convert the action to the joint angles.
    # action = torch.clip(action, -clip_actions, clip_actions).to(self.device)

    action_scaled = action * action_scale
    motor_targets = action_scaled + default_dof_pos

    return np.array(motor_targets.tolist())


def run_example(num_max_steps=_NUM_STEPS):
  """Runs the example.

  Args:
    num_max_steps: Maximum number of steps this example should run for.
  """
  path = '/home/pi/model_500.pt'
  device = 'cpu'

  policy = IsaacGymPolicy(path, device)

  env = env_loader.load()
  env.seed(_ENV_RANDOM_SEED)

  env._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  env._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

  # BUG: the motor limits defined by the robot override those of the motor model here
  # https://github.com/bulletphysics/bullet3/blob/48dc1c45da685c77d3642545c4851b05fb3a1e8b/examples/pybullet/gym/pybullet_envs/minitaur/robots/quadruped_base.py#L131
  # print("env.action_space=",env.action_space)
  obs = env.reset()
  last_control_step = time.time()
  log_dict = {
      't': [],
      'IMU': [],
      'MotorAngle': [],
      'action': []
  }
  if FLAGS.log_to_file:
    f = open("env_log.txt", "wb")
  try:
    env_start_wall = time.time()
    last_spammy_log = 0.0
    for i in range(num_max_steps):
      if FLAGS.realtime or FLAGS.run_on_robot:
        # Sync to real time.
        wall_elapsed = time.time() - env_start_wall
        sim_elapsed = env.env_step_counter * 0.02 # env.env_time_step
        sleep_time = sim_elapsed - wall_elapsed
        if sleep_time > 0:
          time.sleep(sleep_time)
        elif sleep_time < -1 and time.time() - last_spammy_log > 1.0:
          print(f"Cannot keep up with realtime. {-sleep_time:.2f} sec behind, "
                f"sim/wall ratio {(sim_elapsed/wall_elapsed):.2f}.")
          last_spammy_log = time.time()

      delta_time = env.robot.GetTimeSinceReset()
      # 1Hz signal
      phase = delta_time * 1 * np.pi 
      # joint angles corresponding to a standing position
      action = np.array([0, 0.6,-1.2,0, 0.6,-1.2,0, 0.6,-1.2,0, 0.6,-1.2])
      # modulate the default joint angles by a sinusoid to make the robot do pushups
      action[:3] = (np.sin(phase) * 0.6 + 0.8) * action[:3]
      action[9:] = (np.sin(phase) * 0.6 + 0.8) * action[9:]
      # NOTE: We do not fix the loop rate so be careful if using a policy that is solely dependent on time

      before_step_timestamp = time.time()
      obs['command'] = [0.0, -0.3, 0]

      action = policy.step(obs)
      # print('motor_targets', action)
      
      obs, reward, done, _ = env.step(action)
      # print(obs)
      after_step_timestamp = time.time()
      log_dict['t'].append(delta_time)
      log_dict['IMU'].append(obs['IMU'])
      log_dict['MotorAngle'].append(obs['MotorAngle'])
      log_dict['action'].append(action)
      if FLAGS.profile:
        print("loop_dt: ", time.time() - last_control_step, "env.step(): ", after_step_timestamp - before_step_timestamp)

      last_control_step = time.time()
  finally:
    if FLAGS.log_to_file:
      print("logging to file...")
      pickle.dump(log_dict, f)
      f.close()

def main(_):
  _load_config(FLAGS.render)
  run_example()


if __name__ == "__main__":
  app.run(main)


