# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

HIL-SERL (Human-in-the-Loop Sample-Efficient Robotic Reinforcement Learning) is a framework for training robotic manipulation policies using RL with human demonstrations and interventions. The system uses JAX/Flax for RL training with asynchronous actor-learner architecture.

**Key Architecture Pattern**: Training runs with two separate processes:
- **Actor node**: Rolls out policies in the environment, collects transitions, sends data to learner
- **Learner node**: Trains the policy on collected data, sends updated policy back to actor

Communication between nodes happens via [agentlace](https://github.com/youliangtan/agentlace) network protocol.

## Common Commands

### Environment Setup
```bash
# Create conda environment
conda create -n hilserl python=3.10

# Install JAX with GPU support (recommended)
pip install --upgrade "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install serl_launcher
cd serl_launcher
pip install -e .
pip install -r requirements.txt

# Install robot infrastructure (for hardware)
cd serl_robot_infra
pip install -e .
```

### Robot Server (Hardware Only)
```bash
# Start Franka robot server (edit parameters in launch script first)
bash serl_robot_infra/robot_servers/launch_right_server.sh

# Useful curl commands for robot control
curl -X POST http://<SERVER_URL>:5000/getpos_euler  # Get current pose
curl -X POST http://<SERVER_URL>:5000/activate_gripper
curl -X POST http://<SERVER_URL>:5000/close_gripper
curl -X POST http://<SERVER_URL>:5000/jointreset
```

### Training Pipeline

1. **Collect classifier data** (for reward classifiers):
```bash
cd examples
python record_success_fail.py --exp_name <experiment_name> --successes_needed 200
```

2. **Train reward classifier**:
```bash
cd examples/experiments/<experiment_name>
python ../../train_reward_classifier.py --exp_name <experiment_name>
```

3. **Record demonstrations**:
```bash
cd examples/experiments/<experiment_name>
python ../../record_demos.py --exp_name <experiment_name> --successes_needed 20
```

4. **Launch training** (run both in parallel):
```bash
# Terminal 1: Actor
bash run_actor.sh

# Terminal 2: Learner
bash run_learner.sh
```

5. **Evaluate checkpoint**:
```bash
# Edit run_actor.sh to add flags:
# --eval_checkpoint_step=<STEP> --eval_n_trajs=<N>
bash run_actor.sh
```

## Code Architecture

### Core Library Structure (`serl_launcher/`)

- **`agents/`**: RL agent implementations
  - `sac.py`: Standard SAC agent (single-arm, fixed gripper)
  - `sac_hybrid_single.py`: SAC with learned gripper control (single-arm)
  - `sac_hybrid_dual.py`: SAC for dual-arm setups with learned grippers
  - `bc.py`: Behavioral cloning agent

- **`networks/`**: Neural network modules
  - `actor_critic_nets.py`: Policy and critic networks
  - `reward_classifier.py`: Binary reward classifier networks
  - `mlp.py`: MLP building blocks

- **`vision/`**: Visual encoders
  - `resnet_v1.py`: ResNet-10 encoder (scratch or pretrained-frozen)
  - `data_augmentations.py`: Image augmentation functions

- **`data/`**: Replay buffers and data handling
  - `memory_efficient_replay_buffer.py`: Main replay buffer implementation
  - `data_store.py`: Network data transmission interface

- **`wrappers/`**: Gym environment wrappers
  - `serl_obs_wrappers.py`: Observation formatting for SERL agents
  - `chunking.py`: Temporal observation stacking
  - `video_wrapper.py`: Video recording

- **`utils/`**: Training utilities
  - `launcher.py`: Agent factory functions
  - `train_utils.py`: Training helper functions

### Robot Infrastructure (`serl_robot_infra/`)

- **`robot_servers/`**: Flask servers that communicate with ROS controllers
  - `franka_server.py`: Main server for Franka robots
  - HTTP endpoints for pose control, gripper commands, state queries

- **`franka_env/`**: Gym environment for Franka robots
  - Communicates with Flask server via POST requests
  - Wrappers for spacemouse intervention, reward classifiers, relative frames

### Experiment Configuration Pattern

Each task has a folder in `examples/experiments/<task_name>/` containing:

1. **`config.py`**: Defines two config classes
   - `EnvConfig`: Robot/environment settings (server URL, cameras, poses, compliance params)
   - `TrainConfig`: Training hyperparameters, encoder type, setup mode
   - `TrainConfig.get_environment()`: Builds complete wrapped environment

2. **`wrapper.py`**: Custom environment wrapper for task-specific logic

3. **`run_actor.sh`** and **`run_learner.sh`**: Launch scripts with XLA memory settings

### Training Entry Points (`examples/`)

- **`train_rlpd.py`**: Main RLPD training (50/50 demo/online sampling)
  - Keyboard controls: 'f' for failure, 'c' for checkpoint, 'r' for gripper reset, 'p' for pause
  - Saves checkpoints, buffers, demo_buffers, and intervention data periodically

- **`train_bc.py`**: Behavioral cloning pre-training
- **`record_demos.py`**: Collect demos with spacemouse, saves successful episodes only
- **`record_success_fail.py`**: Collect labeled data for reward classifier (hold spacebar for positive labels)
- **`train_reward_classifier.py`**: Train binary image-based reward classifier

### Configuration Mapping

The `examples/experiments/mappings.py` file contains `CONFIG_MAPPING` dict that maps experiment names to their TrainConfig classes. **Always update this when adding new experiments.**

## Important Implementation Details

### Agent Setup Modes

The `setup_mode` in TrainConfig determines agent type:
- `"single-arm-fixed-gripper"`: SACAgent, gripper is always closed
- `"single-arm-learned-gripper"`: SACAgentHybridSingleArm, learns gripper open/close
- `"dual-arm-fixed-gripper"`: SACAgent for bimanual tasks
- `"dual-arm-learned-gripper"`: SACAgentHybridDualArm for bimanual with learned grippers

### Encoder Types

- `"resnet"`: ResNet-10 trained from scratch
- `"resnet-pretrained"`: ResNet-10 with frozen pretrained weights (recommended, faster training)

### Environment Configuration Requirements

When setting up a new task in `config.py`:

1. **Camera configuration**:
   - `REALSENSE_CAMERAS`: Dict with serial numbers and settings
   - `IMAGE_CROP`: Lambda functions to crop images
   - `image_keys`: Cameras for policy training
   - `classifier_keys`: Cameras for reward classifier (can overlap with image_keys)

2. **Robot poses** (collect with `curl -X POST http://<URL>:5000/getpos_euler`):
   - `TARGET_POSE`: Goal pose (e.g., RAM fully inserted)
   - `GRASP_POSE`: Initial grasp pose (if applicable)
   - `RESET_POSE`: Where to reset between episodes
   - `ABS_POSE_LIMIT_LOW/HIGH`: Safety bounding box for exploration

3. **Compliance parameters**:
   - `COMPLIANCE_PARAM`: Stiffness/damping for normal operation
   - `PRECISION_PARAM`: Higher stiffness for precise insertion tasks

### Reward Classifier Pattern

The reward classifier wrapper takes a `reward_func` that:
- Receives observations dict
- Returns binary reward (0 or 1)
- Can use `jnp` operations and multiple camera keys
- Often includes sigmoid threshold (e.g., `sigmoid(classifier(obs)) > 0.85`)

### Buffer and Checkpoint Saving

Training automatically saves:
- `buffer/transitions_<step>.pkl`: All rollout data (full trajectories)
- `demo_buffer/transitions_<step>.pkl`: Only intervention/demo data
- `interventions/transitions_<step>.pkl`: Intervention metadata (t0, t1, policy actions vs human actions)
- Checkpoints saved to `checkpoint_path` every `checkpoint_period` steps

### Human Intervention Flow

1. Actor detects spacemouse input, stores as `info["intervene_action"]`
2. Policy action is overridden with intervention action
3. Transitions are added to both main buffer AND intervention buffer
4. Intervention metadata tracks when human took over and released control

## JAX/Flax Patterns

- All agents are `flax.struct.PyTreeNode` for JAX transformations
- Training state managed by `JaxRLTrainState` with separate optimizers per network
- `@jax.jit` decorators on agent methods for performance
- Use `sharding` for multi-device distribution (currently configured for single GPU)
- Frozen pretrained encoders use `stop_gradient` to prevent updates

## Debugging and Development

- Set `--debug` flag to disable wandb logging
- `eval_checkpoint_step` and `eval_n_trajs` for checkpoint evaluation
- Episode stats logged: intervention_count, intervention_rate, episode_duration, success_rate
- Timer utilities in `utils/timer_utils.py` track performance bottlenecks
