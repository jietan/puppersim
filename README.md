# puppersim
Simulation for DJI Pupper v2 robot

## Installation
Navigate to the puppersim folder in terminal and run
```bash
python setup.py develop
```

## Usage
### Heuristic control
Navigate to the outer puppersim folder and run
```bash
python puppersim/pupper_server.py
```

Clone the the [heuristic controller](https://github.com/stanfordroboticsclub/StanfordQuadruped.git). Navigate to the StanfordQuadruped controller in terminal and run 
```bash
git checkout dji
```
This is so you use the version of code for Pupper V2 rather than the servo-based Pupper V1.

In a separate terminal, navigate to StanfordQuadruped and run 
```bash
python run_djipupper_sim.py
```

Keyboard controls:
* wasd: left joystick --> moves robot forward/back and left/right
* arrow keys: right joystick --> turns robot left/right
* q: L1 --> activates/deactivates robot
* e: R1 --> starts/stops trotting gait
* ijkl: d-pad
* x: X
* square: u
* triangle: t
* circle: c

### Programmatic control
Navigate to the outer puppersim folder and run
```bash
python puppersim/pupper_minimal_server.py
```
In a separate terminal, run
```bash
python puppersim/pupper_example.py --render=False
```


## Training a Gym environment

You can train the pupper using pybullet [envs_v2](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/minitaur/envs_v2) and this [ARS fork](https://github.com/erwincoumans/ars).

```
pip install pybullet arspb ray puppersim
ray start --head
python puppersim/pupper_ars_train.py --rollout_length=200 --policy_type=linear
python puppersim/pupper_ars_run_policy.py --expert_policy_file=data/lin_policy_plus_best_xxx.npz --json_file=data/params.json
```

See a video of a trained policy: https://www.youtube.com/watch?v=JzNsax4M8eg
