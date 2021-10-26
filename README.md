# puppersim
Simulation and Reinforcement Learning for DJI Pupper v2 robot


## Conda setup link
Install [conda](https://docs.conda.io/en/latest/miniconda.html), then
```
conda create --name rl_pupper python=3.7
conda activate rl_pupper
pip install ray arspb
```

## Getting the code ready
```
git clone https://github.com/jietan/puppersim.git
cd puppersim
pip install -e .         (there is a dot at the end)
python3 puppersim/pupper_example.py       (this verifies the installation, you should see pybullet window show up with a pupper in it)
```

If pupper_example.py is running very slowly, try
```bash
python3 puppersim/pupper_minimal_server.py
```
then in a new terminal tab/window
```bash
python3 puppersim/pupper_example.py --render=False
```
This runs the visualizer GUI and simulator as two separate processes.

## Training
```
ray start --head
python3 puppersim/pupper_ars_train.py --rollout_length=200
ray stop (after training is completed)
```


## Test an ARS policy during training (file location may be different)
```
python3 puppersim/pupper_ars_run_policy.py  --expert_policy_file  data/lin_policy_plus_latest.npz  --json_file data/params.json --render
```

## Prerequisites before deployment to Pupper

Set up Avahi (for linux only, one time only per computer)
```
sudo apt install avahi-* (one time)
```
Run the following, you should see ip address of pupper
```
avahi-resolve-host-name raspberrypi.local -4
```
Setup the zero password login for your pupper (Original password on raspberry pi: raspberry)

One time only per computer, run
```
ssh-keygen
```
One time only per pupper, run
* Linux
```
cat ~/.ssh/id_rsa.pub | ssh pi@`avahi-resolve-host-name raspberrypi.local -4 | awk '{print $2}'` 'mkdir .ssh/ && cat >> .ssh/authorized_keys'
```
* MacOs
```
cat ~/.ssh/id_rsa.pub | ssh pi@raspberrypi.local 'mkdir -p .ssh/ && cat >> .ssh/authorized_keys'
```

## Run a pretrained policy on the Pupper robot
* Turn on the Pupper robot, wait for it to complete the calibration motion.
* Connect your laptop with the Pupper using an USB-C cable
* Run the following command on your laptop:
```
./deploy_to_robot.sh python3 puppersim/puppersim/pupper_ars_run_policy.py --expert_policy_file=puppersim/data/lin_policy_plus_latest.npz --json_file=puppersim/data/params.json --run_on_robot
```

## Using the heuristic control
Navigate to the outer puppersim folder and run
```bash
python3 puppersim/pupper_server.py
```

Clone the the [heuristic controller](https://github.com/stanfordroboticsclub/StanfordQuadruped.git):
```bash
git clone https://github.com/stanfordroboticsclub/StanfordQuadruped.git
cd StanfordQuadruped
git checkout dji
```
The `dji` branch is checked out so you can use the version of code for Pupper V2 rather than the servo-based Pupper V1.

In a separate terminal, navigate to StanfordQuadruped and run 
```bash
python3 run_djipupper_sim.py
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
