#!/usr/bin/env python3
"""
Evaluation script for RLPD model trained with diffusion policy initialization.

Usage:
    python eval_rlpd_diffusion.py --exp_name=test_cube --checkpoint_step=48000 --eval_n_trajs=20
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from absl import app, flags
from flax.training import checkpoints
import os

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
flags.DEFINE_integer("checkpoint_step", 48000, "Checkpoint step to evaluate.")
flags.DEFINE_integer("eval_n_trajs", 20, "Number of trajectories to evaluate.")
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


def eval_rlpd(env, agent, sampling_rng):
    """
    Evaluation loop for RLPD policy.

    Args:
        env: Environment to evaluate on
        agent: SAC agent with trained policy
        sampling_rng: JAX random key for sampling

    Returns:
        dict with evaluation metrics
    """
    success_counter = 0
    episode_durations = []
    episode_returns = []

    print_green(f"Starting evaluation for {FLAGS.eval_n_trajs} trajectories...")

    for episode_idx in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        truncated = False
        start_time = time.time()
        episode_return = 0.0
        step_count = 0

        while not done and not truncated:
            # Sample action from policy
            rng, key = jax.random.split(sampling_rng)
            sampling_rng = rng

            # Move observations to device for faster inference
            obs_on_device = jax.tree_map(lambda x: jax.device_put(x), obs)
            actions = agent.sample_actions(observations=obs_on_device, seed=key, argmax=True)
            actions = np.asarray(jax.device_get(actions))

            # Remove batch dimension if present
            if actions.ndim == 2 and actions.shape[0] == 1:
                actions = actions[0]

            # Step environment
            next_obs, reward, done, truncated, info = env.step(actions)
            episode_return += reward
            step_count += 1
            obs = next_obs

        # Record episode metrics
        episode_duration = time.time() - start_time
        episode_durations.append(episode_duration)
        episode_returns.append(episode_return)

        if episode_return > 0:
            success_counter += 1
            status = "✓ SUCCESS"
        else:
            status = "✗ FAILURE"

        print(f"Episode {episode_idx + 1:3d}/{FLAGS.eval_n_trajs} | {status} | "
              f"Return: {episode_return:.1f} | Duration: {episode_duration:.2f}s | "
              f"Steps: {step_count}")

    # Compute statistics
    success_rate = success_counter / FLAGS.eval_n_trajs * 100
    avg_duration = np.mean(episode_durations)
    avg_return = np.mean(episode_returns)

    return {
        "success_rate": success_rate,
        "success_count": success_counter,
        "avg_duration": avg_duration,
        "avg_return": avg_return,
        "episode_durations": episode_durations,
        "episode_returns": episode_returns,
    }


def main(_):
    print_green(f"Evaluating RLPD+Diffusion model for experiment: {FLAGS.exp_name}")
    print_green(f"Checkpoint step: {FLAGS.checkpoint_step}")
    print_green(f"Number of evaluation trajectories: {FLAGS.eval_n_trajs}")

    # Get config
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment {FLAGS.exp_name} not found in CONFIG_MAPPING"
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    # Create environment
    env = config.get_environment(fake_env=False, save_video=FLAGS.save_video, classifier=True)

    # Set up random keys
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # Create agent based on setup mode
    print_green(f"Creating agent with setup_mode: {config.setup_mode}")

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

    # Load checkpoint
    checkpoint_dir = f"test_cube_rlpd_diffusion/checkpoint_{FLAGS.checkpoint_step}"
    if not os.path.exists(checkpoint_dir):
        print_yellow(f"Warning: Checkpoint directory {checkpoint_dir} not found")
        print_yellow(f"Available checkpoints:")
        import glob as glob_module
        checkpoints_list = sorted(glob_module.glob("test_cube_rlpd_diffusion/checkpoint_*"))
        for ckpt in checkpoints_list[-5:]:
            print(f"  - {ckpt}")
        return

    print_green(f"Loading checkpoint from {checkpoint_dir}")
    ckpt = checkpoints.restore_checkpoint(os.path.abspath(checkpoint_dir), agent.state)
    agent = agent.replace(state=ckpt)

    # Put agent on device
    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent))

    print_green("✓ Checkpoint loaded successfully")
    print_green("Starting evaluation...")
    print_green("-" * 80)

    # Run evaluation
    results = eval_rlpd(env, agent, sampling_rng)

    # Print results
    print_green("-" * 80)
    print_green("EVALUATION RESULTS")
    print_green("-" * 80)
    print(f"Success Rate:       {results['success_rate']:.1f}% ({results['success_count']}/{FLAGS.eval_n_trajs})")
    print(f"Average Return:     {results['avg_return']:.2f}")
    print(f"Average Duration:   {results['avg_duration']:.2f}s")
    print(f"Min Duration:       {np.min(results['episode_durations']):.2f}s")
    print(f"Max Duration:       {np.max(results['episode_durations']):.2f}s")
    print_green("-" * 80)

    return results


if __name__ == "__main__":
    app.run(main)
