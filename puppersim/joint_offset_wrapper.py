import numpy as np
import gym
import gin

JOINT_LOWER_BOUND = -0.15
JOINT_UPPER_BOUND = 0.15

@gin.configurable
class JointOffsetWrapperEnv(gym.Wrapper):
  """A wrapped LocomotionGymEnv with a built-in trajectory generator."""

  def __init__(self,
  	       gym_env,	
               joint_lower_bound=JOINT_LOWER_BOUND,
               joint_upper_bound=JOINT_UPPER_BOUND,
               ):
    super().__init__(gym_env)
    self.joint_lower_bound = joint_lower_bound
    self.joint_upper_bound = joint_upper_bound


  def reset(self, initial_motor_angles=None, reset_duration=1.0):
    self.rand_vector = np.random.uniform(self.joint_lower_bound, self.joint_upper_bound, self.env.robot.num_motors)
    if initial_motor_angles is None:
      motor_angles = None
    else: 
      motor_angles = initial_motor_angles + self.rand_vector
    observation = self.env.reset(motor_angles, reset_duration)
    observation['MotorAngle'] = observation['MotorAngle'] - self.rand_vector
    return observation

  def step(self, action):
    observation, reward, done, info = self.env.step(action+self.rand_vector)
    observation['MotorAngle'] = observation['MotorAngle'] - self.rand_vector
    return observation, reward, done, info
    
  def __getattr__(self, name: str):
    # Expose any other attributes of the underlying environment.
    return getattr(self.env, name)
    
