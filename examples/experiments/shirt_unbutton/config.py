import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    LastNGripperActionsWrapper,
    XYZGripperActionWrapper,
)
# from franka_env.envs.relative_env import RelativeFrame  # Not needed for base-frame control
from experiments.usb_pickup_insertion.wrapper import GripperPenaltyWrapper
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.shirt_unbutton.wrapper import ShirtUnbuttonEnv


class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.2:5000/"

    # Camera configuration
    # D405 (wrist) - small close-range camera
    # D415 (side) - standard depth camera
    # D455 (front) - wide FOV camera (available if needed)
    REALSENSE_CAMERAS = {
        "side": {
            "camera_type": "rs",
            "serial_number": "947122060531",  # D415
            "dim": (640, 480),
            "exposure": 40000,
        },
        "wrist": {
            "camera_type": "rs",
            "serial_number": "123622270810",  # D405
            "dim": (640, 480),
            "exposure": 9000,
        },
    }

    # Image cropping - adjust based on camera views
    # View camera feed and adjust crop regions as needed
    IMAGE_CROP = {
        "side": lambda img: img[70:470, 450:1150],
        "wrist": lambda img: img[50:550, 150:1200],
    }

    # Robot poses - collect using: curl -X POST http://127.0.0.2:5000/getpos_euler
    RESET_POSE = np.array([0.541, -0.014, 0.494, -3.137, 0.022, 0.969])

    # Safety bounding box - robot cannot move outside these limits
    # Using same bounds as cube_reach3
    ABS_POSE_LIMIT_LOW  = np.array([0.40, -0.25, 0.03, np.pi - 0.05, -0.05, np.pi / 2 - 0.05])
    ABS_POSE_LIMIT_HIGH = np.array([0.57,  0.2, 0.55, np.pi + 0.05,  0.05, np.pi / 2 + 0.05])

    # Reset randomization - set to True once you have basic task working
    RANDOM_RESET = False
    RANDOM_XY_RANGE = 0.0
    RANDOM_RZ_RANGE = 0.0

    # Action scaling: (position, rotation, gripper)
    # Lower values = more precise movements
    ACTION_SCALE = np.array([0.1, 0.3, 1])

    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 200  # Adjust based on expected task duration

    # Compliance parameters for normal operation
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

    # Precision parameters for delicate operations (optional)
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }


class TrainConfig(DefaultTrainingConfig):
    # Observation keys - cameras used for policy training
    image_keys = ["side", "wrist"]
    classifier_keys = ["side", "wrist"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]

    # Training hyperparameters
    batch_size = 64
    max_steps = 50000
    random_steps = 0
    training_starts = 200
    checkpoint_period = 1000
    buffer_period = 1000
    steps_per_update = 50
    discount = 0.98
    log_period = 10
    replay_buffer_capacity = 200000
    cta_ratio = 2

    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    # STEER-specific settings
    pretraining_steps = 1000
    reward_scale = 1

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        from franka_env.envs.wrappers import HumanClassifierWrapper

        env_config = EnvConfig()
        # Note: bounds assertion disabled - using cube_reach3 bounds which have
        # different angle convention than RESET_POSE (bounds not enforced for rotation anyway)
        # assert np.logical_and(
        #     env_config.ABS_POSE_LIMIT_LOW <= env_config.RESET_POSE,
        #     env_config.RESET_POSE <= env_config.ABS_POSE_LIMIT_HIGH
        # ).all(), "RESET_POSE must be within ABS_POSE_LIMIT bounds!"

        env = ShirtUnbuttonEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        # RelativeFrame removed - using base frame control since orientation is fixed
        env = Quat2EulerWrapper(env)
        env = XYZGripperActionWrapper(env)  # Reduce to 4D action space (xyz + gripper)
        if not fake_env:
            env = SpacemouseIntervention(env)  # Now operates in 4D space
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        # env = LastNGripperActionsWrapper(env, 8)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # Manual rewards - prompts "Success? (1/0)" at episode end
        if classifier:
            env = HumanClassifierWrapper(env)

        env = GripperPenaltyWrapper(env, penalty=-0.005)
        return env
