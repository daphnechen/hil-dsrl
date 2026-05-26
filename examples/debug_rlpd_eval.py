#!/usr/bin/env python3
"""
Debugging script to investigate why RLPD policy achieves 0% success rate.

Systematically checks:
1. Checkpoint loading
2. Agent state validity
3. Environment setup
4. Action sampling
5. Episode execution with logging
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

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
)

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "test_cube", "Experiment name")
flags.DEFINE_integer("checkpoint_step", 48000, "Checkpoint to debug")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_integer("debug_episodes", 2, "Number of episodes to debug")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


def print_blue(x):
    return print("\033[94m {}\033[00m".format(x))


def print_red(x):
    return print("\033[91m {}\033[00m".format(x))


def main(_):
    print_blue("=" * 80)
    print_blue("RLPD EVALUATION DEBUG")
    print_blue("=" * 80)

    # =========================================================================
    # Step 1: Load Configuration
    # =========================================================================
    print_green("\n[Step 1] Loading configuration...")
    try:
        config = CONFIG_MAPPING[FLAGS.exp_name]()
        print(f"  ✓ Config loaded: {FLAGS.exp_name}")
        print(f"    - Setup mode: {config.setup_mode}")
        print(f"    - Encoder type: {config.encoder_type}")
        print(f"    - Max steps: {config.max_steps}")
    except Exception as e:
        print_red(f"  ✗ Failed to load config: {e}")
        return

    # =========================================================================
    # Step 2: Create Environment
    # =========================================================================
    print_green("\n[Step 2] Creating environment...")
    try:
        env = config.get_environment(fake_env=False, save_video=False, classifier=True)
        print(f"  ✓ Environment created")
        print(f"    - Observation space: {env.observation_space}")
        print(f"    - Action space: {env.action_space}")

        # Test reset
        obs, info = env.reset()
        print(f"  ✓ Environment reset successful")
        print(f"    - Observation keys: {list(obs.keys())}")
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                print(f"      {key}: shape {val.shape}, dtype {val.dtype}")
    except Exception as e:
        print_red(f"  ✗ Failed to create environment: {e}")
        return

    # =========================================================================
    # Step 3: Create Agent
    # =========================================================================
    print_green("\n[Step 3] Creating SAC agent...")
    try:
        if config.setup_mode == 'single-arm-learned-gripper':
            agent = make_sac_pixel_agent_hybrid_single_arm(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=config.image_keys,
                encoder_type=config.encoder_type,
                discount=config.discount,
            )
        else:
            agent = make_sac_pixel_agent(
                seed=FLAGS.seed,
                sample_obs=env.observation_space.sample(),
                sample_action=env.action_space.sample(),
                image_keys=config.image_keys,
                encoder_type=config.encoder_type,
                discount=config.discount,
            )
        print(f"  ✓ Agent created")
        print(f"    - Agent type: {type(agent).__name__}")
        print(f"    - Has state: {hasattr(agent, 'state')}")
    except Exception as e:
        print_red(f"  ✗ Failed to create agent: {e}")
        return

    # =========================================================================
    # Step 4: Load Checkpoint
    # =========================================================================
    print_green("\n[Step 4] Loading checkpoint...")
    checkpoint_dir = f"test_cube_rlpd_diffusion/checkpoint_{FLAGS.checkpoint_step}"
    try:
        if not os.path.exists(checkpoint_dir):
            print_red(f"  ✗ Checkpoint directory not found: {checkpoint_dir}")
            return

        print(f"  Loading from: {checkpoint_dir}")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(checkpoint_dir),
            agent.state
        )
        agent = agent.replace(state=ckpt)
        print(f"  ✓ Checkpoint loaded successfully")
        print(f"    - Params keys: {list(agent.state.params.keys())}")
    except Exception as e:
        print_red(f"  ✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Step 5: Place Agent on Device
    # =========================================================================
    print_green("\n[Step 5] Placing agent on device...")
    try:
        agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent))
        print(f"  ✓ Agent placed on device")
    except Exception as e:
        print_red(f"  ✗ Failed to place on device: {e}")
        return

    # =========================================================================
    # Step 6: Test Single Action Sampling
    # =========================================================================
    print_green("\n[Step 6] Testing action sampling...")
    try:
        obs_test, _ = env.reset()
        rng = jax.random.PRNGKey(FLAGS.seed)
        rng, key = jax.random.split(rng)

        obs_on_device = jax.tree_map(lambda x: jax.device_put(x), obs_test)
        print(f"  Sampling action with argmax=False...")
        action_stochastic = agent.sample_actions(
            observations=obs_on_device,
            seed=key,
            argmax=False
        )
        print(f"    - Shape: {action_stochastic.shape}")
        print(f"    - Values: {action_stochastic}")
        print(f"    - Min: {jnp.min(action_stochastic):.4f}, Max: {jnp.max(action_stochastic):.4f}")

        print(f"  Sampling action with argmax=True...")
        rng, key = jax.random.split(rng)
        action_deterministic = agent.sample_actions(
            observations=obs_on_device,
            seed=key,
            argmax=True
        )
        print(f"    - Shape: {action_deterministic.shape}")
        print(f"    - Values: {action_deterministic}")
        print(f"    - Min: {jnp.min(action_deterministic):.4f}, Max: {jnp.max(action_deterministic):.4f}")

        # Check if actions are in valid range
        action_space_low = env.action_space.low
        action_space_high = env.action_space.high
        print(f"\n  Action space bounds:")
        print(f"    - Low: {action_space_low}")
        print(f"    - High: {action_space_high}")
        print(f"  Are actions in bounds? {np.all(action_deterministic >= action_space_low) and np.all(action_deterministic <= action_space_high)}")

    except Exception as e:
        print_red(f"  ✗ Failed to sample actions: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Step 7: Run Debug Episodes
    # =========================================================================
    print_green(f"\n[Step 7] Running {FLAGS.debug_episodes} debug episodes...")
    print_blue("-" * 80)

    rng = jax.random.PRNGKey(FLAGS.seed)

    for ep_idx in range(FLAGS.debug_episodes):
        print(f"\nEpisode {ep_idx + 1}/{FLAGS.debug_episodes}")
        print("-" * 40)

        try:
            obs, info = env.reset()
            print(f"  Reset successful")

            done = False
            truncated = False
            step = 0
            episode_return = 0.0

            while not done and not truncated and step < 100:  # Max 100 steps per episode
                # Sample action
                rng, key = jax.random.split(rng)
                obs_on_device = jax.tree_map(lambda x: jax.device_put(x), obs)

                action = agent.sample_actions(
                    observations=obs_on_device,
                    seed=key,
                    argmax=True
                )
                action = np.asarray(jax.device_get(action))

                # Remove batch dim if present
                if action.ndim == 2 and action.shape[0] == 1:
                    action = action[0]

                # Step environment
                next_obs, reward, done, truncated, info = env.step(action)
                episode_return += reward

                print(f"  Step {step+1:3d} | Action: {action} | Reward: {reward} | Done: {done}")

                obs = next_obs
                step += 1

            print(f"  Episode complete: Return={episode_return}, Steps={step}")
            if episode_return > 0:
                print(f"  ✓ EPISODE SUCCESS")
            else:
                print(f"  ✗ EPISODE FAILURE")

        except Exception as e:
            print_red(f"  ✗ Episode failed: {e}")
            import traceback
            traceback.print_exc()

    print_blue("\n" + "=" * 80)
    print_blue("DEBUG COMPLETE")
    print_blue("=" * 80)


if __name__ == "__main__":
    app.run(main)
