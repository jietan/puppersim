# puppersim
Simulation for DJI Pupper v2 robot

## Usage:

python3 setup.py develop

Then run puppersim/pupper_server.py

In a separate terminal, run the StanfordQuadruped run_djipupper_sim from this [fork](https://github.com/erwincoumans/StanfordQuadruped).

Keyboard controls:
* wasd: left joystick
* arrow keys: right joystick
* q: L1
* e: R1
* ijkl: d-pad
* x: X
* square: u
* triangle: t
* circle: c

## Training a Gym environment

You can train the pupper using pybullet [envs_v2](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/minitaur/envs_v2) and this [ARS fork](https://github.com/erwincoumans/ars).

'''
pip install pybullet arspb ray puppersim
ray start --head
python3 puppersim/pupper_ars_train.py --policy_type=linear
python3 puppersim/pupper_ars_run_policy.py ----expert_policy_file=data/lin_policy_plus_best.npz --json_file=data/params.json
'''

See a video of a trained policy: https://www.youtube.com/watch?v=JzNsax4M8eg
