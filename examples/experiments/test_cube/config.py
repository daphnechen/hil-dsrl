import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.test_cube.wrapper import TestCubeEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"  # Update to your robot server URL

    REALSENSE_CAMERAS = {
        "front": {
            "camera_type": "rs", # Realsense
            "serial_number": "032522250211",
            "dim": (640, 480), # (1280, 720),
            "exposure": 40000,
        },
        "side": {
            "camera_type": "rs", # Realsense
            "serial_number": "947122060531",
            "dim": (1280, 720), # (640, 480),
            "exposure": 40000,
        },
        "wrist": {
            "camera_type": "rs",
            "serial_number": "123622270810", #"123622270802",
            "dim": (1280, 720),
            "exposure": 9000, #13000,
        },
    }

    # IMAGE_CROP = {
    #     "wrist": lambda img: img[50:550, 150:1200],  # TODO: Adjust after viewing camera feed
    # }
    IMAGE_CROP = {
        "front": lambda img: img[180:430, 150:550],  
        "side": lambda img: img[70:470, 450:1150], 
    }

    # TODO: Collect these poses using: curl -X POST http://<SERVER_URL>:5000/getpos_euler
    # Move robot to grasp position on cube, then run curl command
    TARGET_POSE = np.array([0.6, 0.1, 0.1, np.pi, 0, np.pi/2])  # Pose grasping cube

    # Move robot to starting position above workspace, then run curl command
    RESET_POSE = np.array([0.6, 0.1, 0.15, np.pi, 0, np.pi/2])   # Starting pose above cube

    # Safety limits - adjust based on your workspace
    # Make sure these limits keep robot within safe operating area
    ABS_POSE_LIMIT_LOW = RESET_POSE - np.array([0.15, 0.15, 0.08, 0.1, 0.1, 0.2])
    ABS_POSE_LIMIT_HIGH = RESET_POSE + np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.2])

    # Start simple with no randomization
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0

    ACTION_SCALE = np.array([0.05, 0.2, 1])  # (position, rotation, gripper)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100

    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.005,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "side", "wrist"] # use?
    classifier_keys = ["front", "side", "wrist"] # use?
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]

    # For BC/RLPD training
    batch_size = 256
    max_steps = 50000  # RLPD training steps (expected convergence ~40-50k)
    random_steps = 0
    training_starts = 200
    checkpoint_period = 2000
    buffer_period = 1000
    steps_per_update = 50  # Updates per online rollout step
    demo_ratio = 0.5  # 50% demos, 50% online data for RLPD
    discount = 0.97  # Discount factor for RL
    log_period = 10  # Logging frequency
    replay_buffer_capacity = 200000  # Replay buffer size

    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"  # Learn gripper open/close

    pretraining_steps = 5000  # Number of BC pretraining steps on demos
    reward_scale = 1.0  # Scale factor for rewards (for RLPD training)

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        from franka_env.envs.wrappers import HumanClassifierWrapper

        env = TestCubeEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # For simple BC demo collection, use HumanClassifierWrapper
        # This prompts for success (1/0) after each episode, avoiding the need to train a classifier
        if classifier:
            env = HumanClassifierWrapper(env)

        # Add at the end of get_environment:
        original_step = env.step
        def debug_step(action):
            obs, rew, done, truncated, info = original_step(action)
            if "left" in info or "right" in info:
                print(f"Button states - left: {info.get('left')}, right: {info.get('right')}")
            return obs, rew, done, truncated, info
        env.step = debug_step

        return env
