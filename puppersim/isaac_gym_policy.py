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
import keyboard_utils
from JoystickInterface import JoystickInterface
import os


flags.DEFINE_bool("render", False, "Whether to render the example.")
flags.DEFINE_bool("profile", False, "Whether to print timing results for different parts of the code.")
flags.DEFINE_bool("run_on_robot", False, "Whether to run on robot or in simulation.")
flags.DEFINE_bool("log_to_file", False, "Whether to log data to the disk.")
flags.DEFINE_bool("realtime", False, "Run at realtime.")
flags.DEFINE_string("checkpoint", '/home/pi/nov_6_model1.pt', 'Path to the checkpoint')

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

def _update_speed_from_kb(kb, lin_speed, ang_speed):
  """Updates the controller behavior parameters."""
  if kb.is_keyboard_hit():
    c = kb.get_input_character()
    if c == "w":
      lin_speed += np.array((0.0, -0.1))
    if c == "s":
      lin_speed += np.array((0.0, 0.1))
    if c == "q":
      ang_speed += 0.2
    if c == "e":
      ang_speed += -0.2
    if c == "a":
      lin_speed += np.array((0.1, 0.0))
    if c == "d":
      lin_speed += np.array((-0.1, 0.0))
    if c == "r":
      lin_speed = np.array([0.0, 0.0])
      ang_speed = 0.0

    lin_speed[0] = np.clip(lin_speed[0], -1.0, 1.5)
    lin_speed[1] = np.clip(lin_speed[1], -0.8, 0.8)
    ang_speed = np.clip(ang_speed, -1.2, 1.2)
  return lin_speed, ang_speed

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
    self.default_dof_pos = np.array([-0.2, 0.5, -1.2,
                                     0.2, 0.5, -1.2,
                                     -0.2, 0.5, -1.2,
                                     0.2, 0.5, -1.2])

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
    new_obs =  torch.cat([rp[[1]], rp[[0]], rp_dot[[1]], rp_dot[[0]], motor_angle, command, last_action])
    #new_obs = torch.cat([rp, rp_dot, motor_angle, command, last_action])
    
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
  path = FLAGS.checkpoint
  device = 'cpu'

  policy = IsaacGymPolicy(path, device)

  #keyboard_control = keyboard_utils.KeyboardInput()
  joystick_control = JoystickInterface(config=None)
  lin_speed = np.array([0.0, 0.0])
  ang_speed = 0.0

  env = env_loader.load()
  env.seed(_ENV_RANDOM_SEED)

  env._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  env._pybullet_client.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

  # BUG: the motor limits defined by the robot override those of the motor model here
  # https://github.com/bulletphysics/bullet3/blob/48dc1c45da685c77d3642545c4851b05fb3a1e8b/examples/pybullet/gym/pybullet_envs/minitaur/robots/quadruped_base.py#L131
  # print("env.action_space=",env.action_space)
  obs = env.reset()
  input("Press enter to start")
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
        sim_elapsed = (i+1) * 0.03 # env.env_time_step
        sleep_time = sim_elapsed - wall_elapsed
        if sleep_time > 0:
          time.sleep(sleep_time)
        elif sleep_time < -1 and time.time() - last_spammy_log > 1.0:
          print(f"Cannot keep up with realtime. {-sleep_time:.2f} sec behind, "
                f"sim/wall ratio {(sim_elapsed/wall_elapsed):.2f}.")
          last_spammy_log = time.time()

      before_step_timestamp = time.time()

      #lin_speed, ang_speed = _update_speed_from_kb(
      #          keyboard_control, lin_speed, ang_speed)
      #obs['command'] = [lin_speed[0], lin_speed[1], ang_speed]
      joystick_command = joystick_control.get_command(None)
      obs['command'] = [-joystick_command['horizontal_velocity'][1], -joystick_command['horizontal_velocity'][0], -joystick_command['yaw_rate']]

      action = policy.step(obs)
      #print('motor_targets', action)
      #print('motor angles', env.robot.motor_angles) 
      obs, reward, done, _ = env.step(action)
      # print(obs)
      after_step_timestamp = time.time()
      log_dict['IMU'].append(obs['IMU'])
      log_dict['MotorAngle'].append(obs['MotorAngle'])
      log_dict['action'].append(action)
      if FLAGS.profile:
        print("loop_dt: ", time.time() - last_control_step, "env.step(): ", after_step_timestamp - before_step_timestamp)

      last_control_step = time.time()
  finally:
    env.robot.set_pose(np.array([ 0.01839886,  1.55314338, -2.69348054, -0.02087028,  1.55497551, -2.69269209, 0.01893148,  1.56903708, -2.68736582, -0.01852664,  1.5660969,  -2.68740849]), duration=3.0)
    if FLAGS.log_to_file:
      print("logging to file...")
      pickle.dump(log_dict, f)
      f.close()

def main(_):
  _load_config(FLAGS.render)
  run_example()


if __name__ == "__main__":
  app.run(main)

