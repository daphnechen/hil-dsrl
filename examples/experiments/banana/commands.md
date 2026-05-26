# Shirt Unbutton Experiment Commands

## 0. Robot Server (Terminal 1 - run first)
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/serl_robot_infra/robot_servers
bash launch_right_server.sh
```

## 1. Demo Collection (Terminal 2)
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/examples
sudo python record_demos.py --exp_name=shirt_unbutton --successes_needed=10
```
- Use spacemouse to control robot
- Press **'s'** to mark successful trajectory
- Press **'f'** to mark failure and reset early
- Demos saved to: `./demo_data/shirt_unbutton_10_demos_<timestamp>.pkl`

---

## 2A. Training with STEER

### Terminal 2 - Learner
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/shirt_unbutton
./run_steer_learner.sh --demo_path=../../demo_data/shirt_unbutton_10_demos.pkl
```

### Terminal 3 - Actor
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/shirt_unbutton
./run_steer_actor.sh
```

---

## 2B. Training with HG-DAgger

### Terminal 2 - Learner
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/shirt_unbutton
./run_hgdagger_learner.sh --demo_path=../../demo_data/shirt_unbutton_10_demos.pkl
```

### Terminal 3 - Actor
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/shirt_unbutton
./run_hgdagger_actor.sh
```

---

## 3. Evaluation

### STEER Eval
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/shirt_unbutton
./run_steer_actor.sh --eval_checkpoint_step=<STEP> --eval_n_trajs=10
```

### HG-DAgger Eval
```bash
cd /home/daphne/Desktop/jax-hitl-hil-serl/examples/experiments/shirt_unbutton
./run_hgdagger_actor.sh --eval_checkpoint_step=<STEP> --eval_n_trajs=10
```

Replace `<STEP>` with checkpoint step (e.g., 5000, 10000, etc.)

---

## Keyboard Commands (Actor)
- **'s'** - Mark success
- **'f'** - Mark failure
- **'p'** - Pause training
- **'r'** - Reset gripper
- **'o'** - Open gripper
- **'t'** - Close gripper

---

## Method Comparison

| Method | Description |
|--------|-------------|
| **STEER** | BC on interventions + RL (SAC). Uses fixed temperature=0.005. |
| **HG-DAgger** | BC only on demos + interventions. No RL updates. |

---

## Hyperparameters Summary

| Parameter | Value |
|-----------|-------|
| Action space | 4D (xyz + gripper) |
| batch_size | 64 |
| discount | 0.98 |
| max_steps | 50,000 |
| temperature (STEER) | 0.005 (fixed) |
| encoder | resnet-pretrained |
| Reset XYZ randomization | ±2cm |
| Reset time | 9s |
| Episode length | 200 steps |
