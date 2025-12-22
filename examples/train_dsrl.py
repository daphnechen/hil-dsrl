#!/usr/bin/env python3
"""
DSRL Training Script - Train a steering policy on top of a frozen diffusion policy.

This implements the DSRL algorithm from "Diffusion Steering via Reinforcement Learning"
for sample-efficient fine-tuning of diffusion policies using human-in-the-loop RL.
"""

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted
from pynput import keyboard
import requests

from serl_launcher.agents.continuous.dsrl import DSRLAgent
from serl_launcher.agents.continuous.bc_diffusion import BCDiffusionAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import make_trainer_config, make_wandb_logger
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("diffusion_checkpoint", None, "Path to pre-trained diffusion checkpoint.")
flags.DEFINE_string("checkpoint_path", None, "Path to save DSRL checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean("debug", False, "Debug mode (disables wandb logging).")

# DSRL-specific flags
flags.DEFINE_integer("action_horizon", 1, "Action horizon for diffusion policy.")
flags.DEFINE_integer("num_train_timesteps", 20, "Number of diffusion timesteps during training.")
flags.DEFINE_integer("num_inference_timesteps", 8, "Number of diffusion timesteps during inference.")
flags.DEFINE_integer("critic_ensemble_size", 1, "Number of critics in ensemble.")

devices = jax.local_devices()[1]
num_devices = 1
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

failure_key = False
checkpoint_key = False
pause_key = False


def on_press(key):
    global failure_key
    global checkpoint_key
    global pause_key
    try:
        if str(key) == "'f'":
            failure_key = True
        elif str(key) == "'c'":
            checkpoint_key = True
        elif str(key) == "'r'":
            print_yellow("reset")
            requests.post("http://localhost:5000/reset_gripper")
        elif str(key) == "'t'":
            print_yellow("close gripper for reset")
            requests.post("http://localhost:5000/close_gripper")
        elif str(key) == "'p'":
            print_yellow("pause")
            pause_key = not pause_key
    except AttributeError:
        print("error")
        pass


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    Actor loop: Interact with environment using DSRL steering policy.

    DSRL Action Sampling:
    1. π^W(s) → w_steer (steering noise from learned policy)
    2. π_dp^W(s, w_steer) → a (action from diffusion with steered noise)
    """
    global failure_key, pause_key

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Separate data store for interventions
    intvn_client = TrainerClient(
        "actor_env_intervene",
        FLAGS.ip,
        make_trainer_config(),
        intvn_data_store,
        wait_for_server=True,
    )

    # Evaluation mode
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()

            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.tree_map(lambda x: x[None, ...], obs),
                    seed=key,
                    argmax=True,  # Use mode for evaluation
                )
                actions = np.asarray(jax.device_get(actions[0]))
                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs
                done = done or truncated

            episode_success = bool(info["success"])
            episode_length = info["episode"]["l"]
            print_green(
                f"Episode {episode}: Success: {episode_success}, Length: {episode_length}"
            )
            success_counter += int(episode_success)
            time_list.append(time.time() - start_time)

        print_green(
            f"Success rate: {success_counter / FLAGS.eval_n_trajs}, "
            f"Average time: {np.mean(time_list):.2f}"
        )
        return

    # Training mode
    obs, _ = env.reset()
    done = False

    # Keyboard listener for interventions
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while True:
        if pause_key:
            time.sleep(0.5)
            continue

        timer = Timer()
        sampling_rng, key = jax.random.split(sampling_rng)

        # DSRL steering: π^W → w → π_dp^W → a
        actions = np.asarray(
            jax.device_get(
                agent.sample_actions(
                    observations=jax.tree_map(lambda x: x[None, ...], obs),
                    seed=key,
                )
            )[0]
        )
        timer.tick("sample actions")

        # Check for human intervention
        intervene_action = info.get("intervene_action", None) if not done else None

        # Override with intervention if provided
        if intervene_action is not None:
            actions = intervene_action

        # Execute action
        next_obs, reward, done, truncated, info = env.step(actions)
        timer.tick("step")

        # Add transition to replay buffer
        transition = dict(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            rewards=reward,
            masks=1.0 - done,
            dones=done,
        )

        data_store.insert(transition)

        # Also add to intervention buffer if human intervened
        if intervene_action is not None:
            intvn_data_store.insert(transition)

        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
            done = False

            # Manual failure labeling
            if failure_key:
                print_yellow("Failure label registered!")
                failure_key = False

        timer.tick("finish")

        if checkpoint_key:
            print_yellow("Saving checkpoint...")
            client.request("save_checkpoint")
            checkpoint_key = False


def learner(
    rng,
    agent,
    replay_buffer,
    demo_buffer,
    wandb_logger=None,
):
    """
    Learner loop: Train DSRL agent (Q^A, Q^W, π^W).

    DSRL Update (Algorithm 1):
    1. Update Q^A with TD learning: L_A = (Q^A(s,a) - (r + γ Q^A(s', a')))^2
    2. Update Q^W by distilling Q^A: L_W = (Q^W(s,w) - Q^A(s, π_dp^W(s,w)))^2
    3. Update π^W to maximize Q^W: L_π = -E[Q^W(s, π^W(s))]
    """
    # Training config
    from serl_launcher.utils.launcher import make_trainer_config

    config = make_trainer_config()

    # Create trainer server
    trainer_server = TrainerServer(make_trainer_config())
    trainer_server.register_data_store("actor_env", replay_buffer)
    trainer_server.register_data_store("actor_env_intervene", demo_buffer)
    trainer_server.start(threaded=True)

    # Replay ratio: 1 grad step per env step
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # Demo buffer iterator (for RLPD-style 50/50 mixing)
    if demo_buffer is not None:
        demo_buffer_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size // 2,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )
    else:
        demo_buffer_iterator = None

    # Checkpoint saving callback
    def save_checkpoint(step):
        if FLAGS.checkpoint_path is not None:
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path,
                agent.state,
                step=step,
                keep=10,
                overwrite=True,
            )
            print_green(f"Saved checkpoint at step {step}")

    trainer_server.register_callback("save_checkpoint", save_checkpoint)

    # Training loop
    timer = Timer()
    for step in tqdm.tqdm(range(config.max_steps), dynamic_ncols=True):
        timer.tick("total")

        # Sample batch (50% online, 50% demos for RLPD)
        batch = next(replay_buffer_iterator)
        timer.tick("sample_replay_buffer")

        if demo_buffer_iterator is not None:
            demo_batch = next(demo_buffer_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            timer.tick("sample_demo_buffer")

        # DSRL update
        agent, update_info = agent.update(batch)
        timer.tick("train")

        # Logging
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        # Checkpoint saving
        if step > 0 and step % config.checkpoint_period == 0:
            save_checkpoint(step)

        timer.tock("total")


def main(_):
    # Load experiment config
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment {FLAGS.exp_name} not found in CONFIG_MAPPING"
    config_mod = CONFIG_MAPPING[FLAGS.exp_name]
    config = config_mod.TrainConfig()

    # Determine devices
    assert FLAGS.learner + FLAGS.actor == 1, "Either learner or actor must be specified"

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # Create environment
    env = config.get_environment(
        fake_env=FLAGS.learner,
        classifier=not FLAGS.learner,
        save_video=FLAGS.save_video,
    )
    env = RecordEpisodeStatistics(env)

    # Get observation and action shapes
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    action_dim = sample_action.shape[-1]

    print_green("=" * 80)
    print_green(f"DSRL Training: {FLAGS.exp_name}")
    print_green("=" * 80)
    print_green(f"Observation keys: {list(sample_obs.keys())}")
    print_green(f"Action dim: {action_dim}")
    print_green(f"Mode: {'Learner' if FLAGS.learner else 'Actor'}")
    print_green("=" * 80)

    ##################################################
    # Load Pre-trained Diffusion Policy
    ##################################################
    print_green("Loading pre-trained diffusion policy...")

    assert FLAGS.diffusion_checkpoint is not None, "Must provide --diffusion_checkpoint"
    assert os.path.exists(FLAGS.diffusion_checkpoint), f"Diffusion checkpoint not found: {FLAGS.diffusion_checkpoint}"

    # Create diffusion agent
    diffusion_agent = BCDiffusionAgent.create(
        rng=rng,
        observations=sample_obs,
        actions=sample_action,
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        action_horizon=FLAGS.action_horizon,
        num_train_timesteps=FLAGS.num_train_timesteps,
        num_inference_timesteps=FLAGS.num_inference_timesteps,
    )

    # Load checkpoint
    diffusion_agent = diffusion_agent.replace(
        state=checkpoints.restore_checkpoint(
            FLAGS.diffusion_checkpoint,
            diffusion_agent.state,
        )
    )

    print_green("✓ Diffusion policy loaded successfully!")
    print_green(f"  Checkpoint: {FLAGS.diffusion_checkpoint}")
    print_green(f"  Action horizon: {FLAGS.action_horizon}")
    print_green(f"  Inference timesteps: {FLAGS.num_inference_timesteps}")

    ##################################################
    # Create DSRL Agent
    ##################################################
    print_green("Creating DSRL agent...")

    # Freeze diffusion policy
    frozen_diffusion = jax.lax.stop_gradient(diffusion_agent)
    noise_dim = action_dim * FLAGS.action_horizon

    # Create DSRL agent
    rng, agent_rng = jax.random.split(rng)
    agent = DSRLAgent.create(
        rng=agent_rng,
        observations=sample_obs,
        actions=sample_action,
        diffusion_policy=frozen_diffusion,
        encoder_type=config.encoder_type,
        image_keys=config.image_keys,
        noise_dim=noise_dim,
        critic_ensemble_size=FLAGS.critic_ensemble_size,
        discount=0.99,
        soft_target_update_rate=0.005,
        learning_rate=3e-4,
    )

    print_green("✓ DSRL agent created successfully!")
    print_green(f"  Networks: Q^A, Q^W, π^W")
    print_green(f"  Noise dim: {noise_dim}")
    print_green(f"  Critic ensemble size: {FLAGS.critic_ensemble_size}")

    ##################################################
    # Load Demonstrations
    ##################################################
    demo_buffer = None
    if FLAGS.demo_path:
        print_green(f"Loading demonstrations from {FLAGS.demo_path}...")

        demo_transitions = []
        for path in FLAGS.demo_path:
            demo_files = natsorted(glob.glob(path))
            for demo_file in demo_files:
                with open(demo_file, "rb") as f:
                    demo_data = pkl.load(f)
                    demo_transitions.extend(demo_data)

        print_green(f"✓ Loaded {len(demo_transitions)} demo transitions")

        # Create demo buffer
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=len(demo_transitions),
            image_keys=config.image_keys,
        )

        # Add demos to buffer
        for transition in demo_transitions:
            demo_buffer.insert(transition)

        print_green(f"✓ Demo buffer created with {len(demo_transitions)} transitions")

    ##################################################
    # Actor or Learner
    ##################################################
    if FLAGS.actor:
        # Actor: Collect data using DSRL steering
        print_green("Starting actor (DSRL steering policy)...")

        # Create replay buffer for online data
        replay_buffer = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        actor(agent, replay_buffer, intvn_data_store, env, sampling_rng)

    elif FLAGS.learner:
        # Learner: Train DSRL (Q^A, Q^W, π^W)
        print_green("Starting learner (DSRL training)...")

        # Create replay buffer
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=200000,
            image_keys=config.image_keys,
        )

        # Setup wandb logging
        wandb_logger = None
        if not FLAGS.debug:
            wandb_logger = make_wandb_logger(
                project="serl_dsrl",
                description=FLAGS.exp_name,
                debug=FLAGS.debug,
            )

        learner(
            rng,
            agent,
            replay_buffer,
            demo_buffer,
            wandb_logger=wandb_logger,
        )


if __name__ == "__main__":
    app.run(main)
