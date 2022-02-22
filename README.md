# Pupper Sim
Simulation and Reinforcement Learning for DJI Pupper v2 Robot


## Conda setup link
Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), then
```
conda create --name rl_pupper python=3.7
conda activate rl_pupper
pip install ray arspb
```

## Getting the code ready
```bash
git clone https://github.com/jietan/puppersim.git
cd puppersim
pip install -e .
```
Then to verify the installation, run
```bash
python3 puppersim/pupper_example.py
```
You should see the PyBullet GUI pop up and see Pupper doing an exercise.

<details>
  <summary>Click for instructions if pupper_example.py is running slowly</summary>

  Stop `pupper_example.py`. Then run
  ```bash
  python3 puppersim/pupper_minimal_server.py
  ```
  then in a new terminal tab/window
  ```bash
  python3 puppersim/pupper_example.py --render=False
  ```
  This runs the visualizer GUI and simulator as two separate processes.
</details>
<br/>

## Training
```bash
python3 puppersim/pupper_ars_train.py --rollout_length=200
```

### Troubleshooting
<details>
<summary>Click to expand</summary>

* **Pybullet hangs when starting training**. Possible issue: You have multiple suspended pybullet clients. Solution: Restart your computer. 
</details>
<br/>

### Protocol for saving policies
<details>
<summary>Click to expand</summary>

If you want to save a policy, create a folder within `puppersim/data` with the type of gait and date, eg `pretrained_trot_1_22_22`. From the `data` folder, copy the following files into the folder you just made.


* The `.npz` policy file you want, e.g. `lin_policy_plus_latest.npz`
* `log.txt`
* `params.json`

From `puppersim/config` also copy the `.gin` file you used to train the robot, e.g. `pupper_pmtg.gin` file into the folder you just made. When you run a policy on the robot, make sure your `pupper_robot_*_.gin` file matches the `pupper_pmtg.gin` file you saved.

Then add a `README.md` in the folder with a brief description of what you did, including your motivation for saving this policy. 
</details>
<br/>

## Test an ARS policy during training (file location may be different)
```
python3 puppersim/pupper_ars_run_policy.py  --expert_policy_file  data/lin_policy_plus_latest.npz  --json_file data/params.json --render
```

## Deployment
### Prerequisites
<details>
<summary>Linux</summary>

Set up Avahi (once per computer)
```
sudo apt install avahi-*
```
Run the following, you should see Pupper's IP address
```
avahi-resolve-host-name raspberrypi.local -4
```
Setup the zero password login for your pupper (once per computer) (original raspberry pi password: raspberry)
```
ssh-keygen
cat ~/.ssh/id_rsa.pub | ssh pi@`avahi-resolve-host-name raspberrypi.local -4 | awk '{print $2}'` 'mkdir .ssh/ && cat >> .ssh/authorized_keys'
```
</details>
<details>
<summary>Mac</summary>

Setup the zero password login for your pupper (only once per computer) (original raspberry pi password: raspberry)

Once per computer, run
```
ssh-keygen
cat ~/.ssh/id_rsa.pub | ssh pi@raspberrypi.local 'mkdir -p .ssh/ && cat >> .ssh/authorized_keys'
```
</details>
<br/>

### Run pretrained policy on Pupper
* Turn on the Pupper robot, wait for it to complete the calibration motion.
* Connect your laptop with the Pupper using an USB-C cable
* Run the following command on your laptop:
```bash
./deploy_to_robot.sh python3 puppersim/puppersim/pupper_ars_run_policy.py --expert_policy_file=puppersim/data/lin_policy_plus_latest.npz --json_file=puppersim/data/params.json --run_on_robot
```

## Simulating the heuristic controller
<details>
  <summary>Click to expand</summary>
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
</details>
