#!/usr/bin/env python3
"""
RLPD training with diffusion policy initialization.

This script extends train_rlpd.py to initialize the SAC policy with weights from
a pre-trained diffusion policy (BCDiffusionAgent). This allows us to test if the
better BC initialization from diffusion leads to faster RL convergence and fewer
human interventions needed.

Key differences from train_rlpd.py:
1. Load pre-trained diffusion BC policy
2. Extract policy network weights and transfer to SAC agent
3. Optional: freeze encoder to match BC training
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

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.agents.continuous.bc_diffusion import BCDiffusionAgent
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
    make_batch_augmentation_func,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("diffusion_bc_checkpoint_path", None, "Path to diffusion BC checkpoint for initialization.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)

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
    The actor loop, which runs when "--actor" is set to True.
    """
    global failure_key, checkpoint_key, pause_key

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )
    # client.publish_network(agent.state.params)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print_green("Actor is connected to server.")
    step = 0
    print(f"actor_env_intvn: {data_store}")
    total_interventions = 0
    import pynput.keyboard
    from gymnasium.core import ActType
    pbar = tqdm.tqdm(total=config.max_steps, initial=0, desc="actor", position=1, leave=True)
    timer = Timer()
    print_green("Actor is starting...")
    interventions = []  # Initialize interventions list
    while True:
        timer.tick("total")
        obs, _ = env.reset()
        done = False
        from_time = time.time()
        running_return = 0.0
        intervention_count = 0
        intervention_steps = 0
        cur_steps = 0
        already_intervened = False
        transitions = []
        demo_transitions = []
        transitions_full_trajs = []
        demo_transitions_full_trajs = []

        while not done and step < config.max_steps:
            with timer.context("inference"):
                rng, key = jax.random.split(sampling_rng)

                # Move observations to device for faster inference
                obs_on_device = jax.tree_map(lambda x: jax.device_put(x), obs)
                actions = agent.sample_actions(observations=obs_on_device, seed=key, argmax=False)
                actions = np.asarray(jax.device_get(actions))

            with timer.context("env_step"):
                next_obs, reward, done, truncated, info = env.step(actions)
                running_return += reward
                step += 1
                cur_steps += 1
                pbar.update(1)

            with timer.context("intervention"):
                if failure_key:
                    print_yellow("reset action")
                    failure_key = False
                    done = True
                    time.sleep(0.5)
                client.update()

            with timer.context("data_store_insert"):
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                # Add grasp_penalty if using learned gripper setup
                if 'grasp_penalty' in info:
                    transition['grasp_penalty'] = info['grasp_penalty']
                else:
                    transition['grasp_penalty'] = 0.0
                data_store.insert(transition)
                transitions.append(copy.deepcopy(transition))
                if already_intervened:
                    intvn_data_store.insert(transition)
                    demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                failure_key = False
                if "episode" not in info:
                    info["episode"] = {}
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps

                total_interventions += intervention_count

                info["episode"]["total_interventions"] = total_interventions
                info["episode"]["intervention_rate"] = total_interventions / (step + 1)
                info["episode"]["current_intervention_rate"] = intervention_steps / cur_steps
                info["episode"]["episode_duration"] = time.time() - from_time
                info["episode"]["success_rate"] = running_return
                info["episode"]["episode_steps"] = cur_steps
                info["episode"]["environment_step"] = step
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                cur_steps = 0

                already_intervened = False
                client.update()
                print("reset start")
                obs, _ = env.reset()
                transitions_full_trajs = transitions
                demo_transitions_full_trajs = demo_transitions
                time.sleep(7.0)
                print("reset end")
                from_time = time.time()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            # dump to pickle file
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            interventions_buffer_path = os.path.join(FLAGS.checkpoint_path, "interventions")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            if not os.path.exists(interventions_buffer_path):
                os.makedirs(interventions_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                fp = os.path.join(buffer_path, f"transitions_{step}.pkl")
                print(f"Dumping {len(transitions_full_trajs)} expert transitions out of {len(transitions)} to {fp} !!!")
                pkl.dump(transitions_full_trajs, f)
                transitions = transitions[len(transitions_full_trajs):]
                transitions_full_trajs = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"),
                "wb"
            ) as f:
                fp = os.path.join(demo_buffer_path, f"transitions_{step}.pkl")
                print(f"Dumping {len(demo_transitions_full_trajs)} expert transitions out of {len(demo_transitions)} to {fp} !!!")
                pkl.dump(demo_transitions_full_trajs, f)
                demo_transitions = demo_transitions[len(demo_transitions_full_trajs):]
                demo_transitions_full_trajs = []
            with open(os.path.join(interventions_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                fp = os.path.join(interventions_buffer_path, f"transitions_{step}.pkl")
                print(f"Dumping {len(interventions)} interventions to {fp}")
                pkl.dump(interventions, f)
                interventions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

        # Exit training if we've reached max steps
        if step >= config.max_steps:
            print_green(f"Training complete! Reached {step} steps (max: {config.max_steps})")
            break


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})


    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step + config.pretraining_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    if step == 0 and config.pretraining_steps > 0:
        epochs = config.pretraining_steps // config.batch_size
        print(f"Pretraining on {len(demo_buffer)} demo steps for {config.pretraining_steps} steps ({epochs} epochs)...")
        for epoch in tqdm.tqdm(range(epochs)):
            batch = demo_buffer.sample(config.batch_size)
            agent, update_info = agent.update(batch, networks_to_update=frozenset({"critic", "grasp_critic", "actor"}))
            wandb_logger.log({'pretraining': update_info}, step=(epoch + 1) * config.batch_size)
        agent = jax.block_until_ready(agent)
        server.publish_network(agent.state.params)
        checkpoints.save_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path), agent.state, step=1, keep=100
        )

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )

    timer = Timer()
    while len(replay_buffer) < config.training_starts:
        time.sleep(1)
        pbar.update(len(replay_buffer) - pbar.n)

    # main training loop
    pbar = tqdm.tqdm(
        total=config.max_steps,
        initial=step,
        desc="training",
        position=0,
        leave=True,
    )
    for step in range(step, config.max_steps):
        pbar.update(1)
        with timer.context("sample_batch"):
            batch = concat_batches(
                demo_buffer.sample(int(config.batch_size * config.demo_ratio)),
                replay_buffer.sample(int(config.batch_size * (1 - config.demo_ratio))),
                axis=0,
            )

        with timer.context("update_networks"):
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        if step > 0 and step % (config.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step + config.pretraining_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step + config.pretraining_steps)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def load_diffusion_bc_and_extract_policy(
    diffusion_checkpoint_path: str,
    env,
    config,
):
    """
    Load a pre-trained diffusion BC policy and extract the policy network weights.

    Returns:
        Tuple of (policy_encoder_params, policy_actor_params) for initialization
    """
    print_green(f"Loading diffusion BC checkpoint from {diffusion_checkpoint_path}")

    # Create diffusion BC agent
    rng = jax.random.PRNGKey(0)
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()

    diffusion_agent: BCDiffusionAgent = BCDiffusionAgent.create(
        rng=rng,
        observations=sample_obs,
        actions=sample_action,
        encoder_type=config.encoder_type,
        image_keys=config.image_keys,
        use_proprio=True,
        action_horizon=1,
        num_train_timesteps=20,
        num_inference_timesteps=8,
        beta_schedule="cosine",
        learning_rate=3e-4,
        augmentation_function=make_batch_augmentation_func(config.image_keys),
    )

    # Load checkpoint
    ckpt = checkpoints.restore_checkpoint(
        diffusion_checkpoint_path,
        diffusion_agent.state,
    )
    diffusion_agent = diffusion_agent.replace(state=ckpt)

    # Extract policy parameters (encoder and actor network)
    # The diffusion policy has: encoder + denoising_network
    # We want to use the encoder and transfer the "policy" aspect to SAC
    policy_params = diffusion_agent.state.params["modules_actor"]

    print_green(f"Extracted policy parameters from diffusion BC checkpoint")
    print_green(f"  Encoder params: {jax.tree_util.tree_structure(policy_params['encoder'])}")
    print_green(f"  Denoising network params: {jax.tree_util.tree_structure(policy_params['denoising_network'])}")

    return policy_params


def initialize_sac_with_diffusion_policy(sac_agent, diffusion_policy_params):
    """
    Initialize SAC agent's policy encoder with weights from diffusion BC policy.

    This transfers the visual encoding learned by diffusion policy to SAC.
    SAC's actor head will be initialized randomly, but the encoder is pre-trained.
    """
    print_yellow("Initializing SAC policy encoder with diffusion BC weights...")

    # Get current SAC params
    current_params = sac_agent.state.params["modules_actor"]

    # Try to transfer encoder weights
    try:
        # Replace encoder in current params with diffusion encoder
        new_params = current_params.copy({
            "encoder": diffusion_policy_params["encoder"]
        })

        # Create new state with updated params
        new_state = sac_agent.state.replace(
            params={"modules_actor": new_params}
        )
        sac_agent = sac_agent.replace(state=new_state)

        print_green("✓ Successfully initialized SAC encoder with diffusion policy weights")
        return sac_agent
    except Exception as e:
        print_yellow(f"Warning: Could not transfer encoder weights: {e}")
        print_yellow("Continuing with random SAC initialization...")
        return sac_agent


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)

    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # Initialize SAC with diffusion policy if checkpoint provided
    if FLAGS.diffusion_bc_checkpoint_path:
        diffusion_params = load_diffusion_bc_and_extract_policy(
            FLAGS.diffusion_bc_checkpoint_path,
            env,
            config,
        )
        agent = initialize_sac_with_diffusion_policy(agent, diffusion_params)

    # Put agent on device
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent)
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=f"{FLAGS.exp_name}_diffusion_init",
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng)
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        num_demos = 0
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    # Add default grasp_penalty if it doesn't exist and we're using grasp_penalty
                    if include_grasp_penalty and 'grasp_penalty' not in transition:
                        transition['grasp_penalty'] = 0.0
                    assert transition['rewards'] < 1 + 1e-6 and transition['rewards'] > -1e-6, f"{transition['rewards']}"
                    num_demos += transition['rewards']
                    transition['rewards'] *= config.reward_scale
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"demo count: {num_demos}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop with diffusion-initialized policy")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng)
        data_store = QueuedDataStore(10000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(10000)

        # actor loop
        print_green("starting actor loop with diffusion-initialized policy")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
