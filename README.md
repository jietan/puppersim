# puppersim
Simulation and Reinforcement Learning for DJI Pupper v2 robot


## Conda setup link
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

## Training
```
ray start --head
python3 puppersim/pupper_ars_train.py --rollout_length=200
ray stop (after training is completed)
```

<<<<<<< HEAD
=======
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
>>>>>>> 690a6ab (add movement instructions)

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
<<<<<<< HEAD
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
=======
pip install pybullet arspb ray puppersim
ray start --head
python puppersim/pupper_ars_train.py --rollout_length=200
python puppersim/pupper_ars_run_policy.py --expert_policy_file=data/lin_policy_plus_best_xxx.npz --json_file=data/params.json
>>>>>>> 738a846 (Make rollout_length=200 in readme)
```

## Run a pretrained policy on the Pupper robot
* Turn on the Pupper robot, wait for it to complete the calibration motion.
* Connect your laptop with the Pupper using an USB-C cable
* Run the following command on your laptop:
```
./deploy_to_robot.sh python3 puppersim/puppersim/pupper_ars_run_policy.py --expert_policy_file=puppersim/data/lin_policy_plus_latest.npz --json_file=puppersim/data/params.json --run_on_robot
```
