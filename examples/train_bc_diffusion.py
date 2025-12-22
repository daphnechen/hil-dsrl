#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import sys
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

# Add examples directory to path if running from root
if os.path.basename(os.getcwd()) != "examples":
    sys.path.insert(0, os.path.join(os.getcwd(), "examples"))

from serl_launcher.agents.continuous.bc_diffusion import BCDiffusionAgent

from serl_launcher.utils.launcher import (
    make_trainer_config,
    make_wandb_logger,
    make_batch_augmentation_func,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("bc_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_integer("train_steps", 20_000, "Number of training steps.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_multi_string("demo_path", None, "Path(s) to demo data files (glob patterns supported).")

# Diffusion-specific flags
flags.DEFINE_integer("action_horizon", 1, "Number of future actions to predict (action chunk size).")
flags.DEFINE_integer("num_train_timesteps", 20, "Number of diffusion timesteps during training.")
flags.DEFINE_integer("num_inference_timesteps", 8, "Number of diffusion timesteps during inference.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


try:
    devices = jax.local_devices()[1:2]  # Use only GPU 1 for BC training
    num_devices = len(devices) if devices else 1
    if devices:
        sharding = jax.sharding.PositionalSharding(devices)
    else:
        sharding = None
except (IndexError, ValueError):
    devices = jax.local_devices()[:1]  # Fallback to first device (CPU)
    num_devices = 1
    sharding = None


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    bc_agent: BCDiffusionAgent,
    sampling_rng,
):
    """
    Evaluation loop for diffusion policy.
    """
    success_counter = 0
    time_list = []
    for episode in range(FLAGS.eval_n_trajs):
        obs, _ = env.reset()
        done = False
        start_time = time.time()
        while not done:
            rng, key = jax.random.split(sampling_rng)

            # Move observations to GPU/device
            obs_on_device = jax.tree_map(lambda x: jax.device_put(x), obs)
            actions = bc_agent.sample_actions(observations=obs_on_device, seed=key)
            actions = np.asarray(jax.device_get(actions))
            # Remove batch dimension if present (shape: (1, 7) -> (7,))
            if actions.ndim == 2 and actions.shape[0] == 1:
                actions = actions[0]
            next_obs, reward, done, truncated, info = env.step(actions)
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    if len(time_list) > 0:
        print(f"average time: {np.mean(time_list)}")


##############################################################################


def train(
    bc_agent: BCDiffusionAgent,
    bc_replay_buffer,
    config: DefaultTrainingConfig,
    wandb_logger=None,
):

    bc_replay_iterator = bc_replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size,
            "pack_obs_and_next_obs": False,
        },
        device=None,  # Use default device placement for single GPU
    )

    # Train diffusion policy
    for step in tqdm.tqdm(
        range(FLAGS.train_steps),
        dynamic_ncols=True,
        desc="bc_diffusion_training",
    ):
        batch = next(bc_replay_iterator)
        bc_agent, bc_update_info = bc_agent.update(batch)
        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log({"bc_diffusion": bc_update_info}, step=step)
        if step > FLAGS.train_steps - 100 and step % 10 == 0:
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.bc_checkpoint_path), bc_agent.state, step=step, keep=5
            )
    print_green("bc diffusion training done and saved checkpoint")


##############################################################################


def make_bc_diffusion_agent(
    seed: int,
    sample_obs,
    sample_action,
    image_keys: tuple,
    encoder_type: str = "resnet-pretrained",
    action_horizon: int = 1,
    num_train_timesteps: int = 20,
    num_inference_timesteps: int = 8,
):
    """
    Create a BCDiffusionAgent.

    Args:
        seed: Random seed
        sample_obs: Sample observation for initialization
        sample_action: Sample action for initialization
        image_keys: Image observation keys
        encoder_type: Type of visual encoder
        action_horizon: Number of future actions to predict
        num_train_timesteps: Diffusion steps during training
        num_inference_timesteps: Diffusion steps during inference

    Returns:
        BCDiffusionAgent instance
    """
    rng = jax.random.PRNGKey(seed)

    agent = BCDiffusionAgent.create(
        rng=rng,
        observations=sample_obs,
        actions=sample_action,
        encoder_type=encoder_type,
        image_keys=image_keys,
        use_proprio=True,
        action_horizon=action_horizon,
        num_train_timesteps=num_train_timesteps,
        num_inference_timesteps=num_inference_timesteps,
        beta_schedule="cosine",
        learning_rate=3e-4,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )

    return agent


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    bc_agent: BCDiffusionAgent = make_bc_diffusion_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        action_horizon=FLAGS.action_horizon,
        num_train_timesteps=FLAGS.num_train_timesteps,
        num_inference_timesteps=FLAGS.num_inference_timesteps,
    )

    # Put agent on device (single GPU, no replication needed for BC)
    bc_agent: BCDiffusionAgent = jax.device_put(
        jax.tree_map(jnp.array, bc_agent)
    )

    if not eval_mode:
        assert not os.path.isdir(
            os.path.join(FLAGS.bc_checkpoint_path, f"checkpoint_{FLAGS.train_steps}")
        )

        bc_replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"{FLAGS.exp_name}_diffusion",
            debug=FLAGS.debug,
        )

        # Load demo data
        if FLAGS.demo_path:
            # Use specified demo paths
            demo_paths = []
            for pattern in FLAGS.demo_path:
                demo_paths.extend(glob.glob(pattern))
        else:
            # Default: load all demos from demo_data/
            demo_paths = glob.glob(os.path.join(os.getcwd(), "demo_data", "*.pkl"))

        assert demo_paths, "No demo files found!"
        print(f"Loading demos from {len(demo_paths)} file(s):")
        for path in demo_paths:
            print(f"  - {path}")

        for path in demo_paths:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if np.linalg.norm(transition['actions']) > 0.0:
                        bc_replay_buffer.insert(transition)
        print(f"bc replay buffer size: {len(bc_replay_buffer)}")

        # learner loop
        print_green("starting diffusion policy training")
        train(
            bc_agent=bc_agent,
            bc_replay_buffer=bc_replay_buffer,
            wandb_logger=wandb_logger,
            config=config,
        )

    else:
        rng = jax.random.PRNGKey(FLAGS.seed)
        sampling_rng = jax.device_put(rng)

        bc_ckpt = checkpoints.restore_checkpoint(
            FLAGS.bc_checkpoint_path,
            bc_agent.state,
        )
        bc_agent = bc_agent.replace(state=bc_ckpt)

        print_green("starting evaluation with diffusion policy")
        eval(
            env=env,
            bc_agent=bc_agent,
            sampling_rng=sampling_rng,
        )


if __name__ == "__main__":
    app.run(main)
