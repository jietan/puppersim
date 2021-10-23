import os

def getPupperSimPath():
  resdir = os.path.join(os.path.dirname(__file__))
  return resdir

try:
  import gym
  from gym.envs.registration import registry, make, spec
  def register(id, *args, **kvargs):
    if id in registry.env_specs:
      return
    else:
      return gym.envs.registration.register(id, *args, **kvargs)

  register(
    id='PupperGymEnv-v0',
    entry_point='puppersim.pupper_gym_env:PupperGymEnv',
    max_episode_steps=150,
    reward_threshold=5.0,
  )

  def getList():
    envs = [spec.id for spec in gym.envs.registry.all() if spec.id.find('Pupper') >= 0]
    return envs

  
except Exception as e:
  pass 


