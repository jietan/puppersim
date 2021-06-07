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

Todo: see pupper_example.py and the [Minitaur/Laikago environments](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/minitaur/envs_v2) as example.
