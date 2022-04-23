# Reacher instructions
## Training
```bash
conda activate rl_pupper
cd puppersim/reacher
python reacher_ars_train.py --policy_type=nn
```
### Available flags
* ``--policy_type={nn or linear}``. Choose neural network or linear policy. Default is linear. Pass ``--policy_type=nn`` for neural network policy.
* ``--rollout_length={integer}``. Number of timesteps for each rollout. Each timestep is 1/240 seconds long. Default is 50.
* ``--n_iter={integer}``. Number of iterations to run ARS for. Default is 1000.

## Running policy
```bash
conda activate rl_pupper
cd puppersim/reacher 
python reacher_ars_run_policy.py  --expert_policy_file  data/lin_policy_plus_latest.npz  --json_file data/params.json --render --realtime
```

### Available flags
* ``--render_meshes`` to show the realistic robot model. By default the simulator will only show a simple model of the robot.
* ``--rollout_length`` number of timesteps (dt=1/240) per rollout. The target will change after each rollout. Default is 200.