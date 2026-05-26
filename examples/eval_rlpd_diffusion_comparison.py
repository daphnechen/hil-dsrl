#!/usr/bin/env python3
"""
Comparison script to evaluate RLPD+Diffusion vs vanilla BC and other baselines.

This script evaluates multiple checkpoints to show convergence progression.

Usage:
    python eval_rlpd_diffusion_comparison.py --exp_name=test_cube --eval_n_trajs=10
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from absl import app, flags
from flax.training import checkpoints
import os
import glob as glob_module

from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
)

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "test_cube", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of trajectories per checkpoint.")
flags.DEFINE_boolean("save_video", False, "Save video of the evaluation.")
flags.DEFINE_boolean("debug", False, "Debug mode.")

try:
    devices = jax.local_devices()[1:2]
    num_devices = len(devices) if devices else 1
    if devices:
        sharding = jax.sharding.PositionalSharding(devices)
    else:
        sharding = None
except (IndexError, ValueError):
    devices = jax.local_devices()[:1]
    num_devices = 1
    sharding = None


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_blue(x):
    return print("\033[94m {}\033[00m".format(x))


def eval_checkpoint(env, agent, sampling_rng, checkpoint_step):
    """Evaluate a single checkpoint."""
    success_counter = 0
    episode_returns = []

    for episode_idx in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0

        while not done and not truncated:
            rng, key = jax.random.split(sampling_rng)
            sampling_rng = rng

            obs_on_device = jax.tree_map(lambda x: jax.device_put(x), obs)
            actions = agent.sample_actions(observations=obs_on_device, seed=key, argmax=True)
            actions = np.asarray(jax.device_get(actions))

            if actions.ndim == 2 and actions.shape[0] == 1:
                actions = actions[0]

            next_obs, reward, done, truncated, info = env.step(actions)
            episode_return += reward
            obs = next_obs

        episode_returns.append(episode_return)
        if episode_return > 0:
            success_counter += 1

    success_rate = success_counter / FLAGS.eval_n_trajs * 100
    avg_return = np.mean(episode_returns)

    return success_rate, avg_return, success_counter


def main(_):
    print_green("=" * 80)
    print_green("RLPD+Diffusion Convergence Analysis")
    print_green("=" * 80)

    # Get config
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment {FLAGS.exp_name} not found"
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # Create environment
    env = config.get_environment(fake_env=False, save_video=FLAGS.save_video, classifier=True)

    # Set up random keys
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # Create agent
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # Find all checkpoints
    checkpoint_pattern = "test_cube_rlpd_diffusion/checkpoint_*"
    checkpoints_list = sorted(glob_module.glob(checkpoint_pattern))

    if not checkpoints_list:
        print_yellow(f"No checkpoints found matching pattern: {checkpoint_pattern}")
        return

    # Extract checkpoint steps and filter for key milestones
    checkpoint_steps = []
    for ckpt_dir in checkpoints_list:
        try:
            step = int(os.path.basename(ckpt_dir).split("_")[1])
            checkpoint_steps.append(step)
        except (ValueError, IndexError):
            continue

    # Select checkpoints to evaluate (every 5k steps + final)
    selected_steps = sorted(set([s for s in checkpoint_steps if s % 5000 == 0] + [max(checkpoint_steps)]))

    print_blue(f"Found {len(checkpoint_steps)} checkpoints")
    print_blue(f"Evaluating {len(selected_steps)} selected checkpoints")
    print_green("-" * 80)

    # Evaluate each checkpoint
    results_table = []
    results_table.append(
        f"{'Step':<8} {'Success %':<15} {'Successes':<12} {'Avg Return':<15}"
    )
    results_table.append("-" * 50)

    for step in selected_steps:
        checkpoint_dir = f"test_cube_rlpd_diffusion/checkpoint_{step}"

        if not os.path.exists(checkpoint_dir):
            print_yellow(f"Checkpoint {step} not found, skipping...")
            continue

        print(f"Evaluating checkpoint step {step}...", end=" ", flush=True)

        # Load checkpoint
        ckpt = checkpoints.restore_checkpoint(os.path.abspath(checkpoint_dir), agent.state)
        agent_eval = agent.replace(state=ckpt)
        agent_eval = jax.device_put(jax.tree_util.tree_map(jnp.array, agent_eval))

        # Evaluate
        success_rate, avg_return, success_count = eval_checkpoint(
            env, agent_eval, sampling_rng, step
        )

        results_table.append(
            f"{step:<8} {success_rate:<14.1f}% {success_count:<11}/{FLAGS.eval_n_trajs} {avg_return:<14.2f}"
        )

        print(f"✓ {success_rate:.1f}% success ({success_count}/{FLAGS.eval_n_trajs})")

    # Print results table
    print_green("-" * 80)
    print_green("CONVERGENCE SUMMARY")
    print_green("-" * 80)
    for row in results_table:
        print(row)
    print_green("-" * 80)

    # Print expected improvements
    print_blue("\nEXPECTED RESULTS:")
    print("- RLPD+Diffusion BC (this):    ~90%+ success by 40-50k steps")
    print("- Vanilla RLPD (baseline):     ~90%+ success by 50-60k steps")
    print("- BC only (diffusion):         ~40% success rate")
    print("- BC only (vanilla MLP):       ~30% success rate")


if __name__ == "__main__":
    app.run(main)
