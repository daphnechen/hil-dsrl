#!/usr/bin/env python3

import glob
import time
# import jax
# import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
# from flax.training import checkpoints
import os
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

# from serl_launcher.agents.continuous.bc import BCAgent

# from serl_launcher.utils.launcher import (
#     make_bc_agent,
#     make_trainer_config,
#     make_wandb_logger,
# )
# from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
import franka_sim.envs.utils as utils

from experiments.mappings import CONFIG_MAPPING
from experiments.config import DefaultTrainingConfig
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("diffusion_checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_bool("save_video", False, "Save video of the evaluation.")
flags.DEFINE_integer("action_horizon", 4, "Action horizon for diffusion policy.")


flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


# devices = jax.local_devices()
# num_devices = len(devices)
# sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


##############################################################################

def eval(
    env,
    diffusion_policy
):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    success_counter = 0
    time_list = []
    buffer = []
    for episode in range(FLAGS.eval_n_trajs):
        traj = []
        obs, _ = env.reset()
        # breakpoint()
        # diffusion_policy.add_obs(obs)
        diffusion_policy.clear_obs()
        done = False
        start_time = time.time()
        while not done:
            # rng, key = jax.random.split(sampling_rng)

            # actions = bc_agent.sample_actions(observations=obs, seed=key)
            # actions = np.asarray(jax.device_get(actions))
            obs = {k:v[0] for k,v in obs.items()}
            actions = diffusion_policy.get_action(obs)
            actions = np.array(actions)


            for a_t in range(FLAGS.action_horizon):

            # if actions[-1] < -0.7:
            #     actions[-1] = -0.9
            # elif actions[-1] > 0.7:
            #     actions[-1] = 0.9
            # else:
            #     actions[-1] = 0.0
                action = actions[a_t]
                # action[-1] = int(action[-1] > 0.5)
                print(f"grip: ", action[-1])
                next_obs, reward, done, truncated, info = env.step(action)
            # diffusion_policy.add_obs(next_obs)
            # traj.append([obs, action, ])
            obs = next_obs
            if done:
                if reward:
                    dt = time.time() - start_time
                    time_list.append(dt)
                    print(dt)
                success_counter += reward
                print(reward)
                print(f"{success_counter}/{episode + 1}")
                env.reset()
                action = np.zeros_like(action)
                action[-1] = 1.0
                env.step(action)
                time.sleep(5.0)
                # diffusion_policy.reset()
                diffusion_policy.clear_obs()
                # env.reset()

    print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
    print(f"average time: {np.mean(time_list)}")


##############################################################################


def main(_):
    config: DefaultTrainingConfig = CONFIG_MAPPING[FLAGS.exp_name]()

    # assert config.batch_size % num_devices == 0
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    eval_mode = FLAGS.eval_n_trajs > 0
    env = config.get_environment(
        fake_env=not eval_mode,
        save_video=FLAGS.save_video,
        classifier=True,
        # eval_env=True
    )
    env = RecordEpisodeStatistics(env)

    from diffusion_policy.policy import DiffusionPolicy
    policy = DiffusionPolicy.load(FLAGS.diffusion_checkpoint_path)

    # # replicate agent across devices
    # # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    # bc_agent: BCAgent = utils.device_put(
    #     jax.tree_map(jnp.array, bc_agent), sharding.replicate()
    # )


    print_green("starting actor loop")
    eval(
        env=env,
        diffusion_policy=policy
    )


if __name__ == "__main__":
    app.run(main)
