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

    REALSENSE_CAMERAS = {
        "wrist": {
            "camera_type": "zed",
            "serial_number": "16744838",      # ZED-M
            "dim": (1280, 720),
            "exposure": 9000,
        },
    }

    # Image cropping - adjust based on camera views
    IMAGE_CROP = {
        "wrist": lambda img: img[:, 200:],    # ZED-M wrist crop (mirrors pen_in_bowl)
    }

    # Robot poses - collect using: curl -X POST http://127.0.0.2:5000/getpos_euler
    # Diagonal gripper orientation (rx=-2.582, ry=-0.026, rz=0.122) for easier grasping
    RESET_POSE = np.array([0.584, -0.202, 0.487, -2.582, -0.026, 0.122])

    # Safety bounding box - disabled (very wide limits)
    ABS_POSE_LIMIT_LOW  = np.array([-10.0, -10.0, -10.0, -np.pi, -np.pi, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([ 10.0,  10.0,  10.0,  np.pi,  np.pi,  np.pi])

    # Reset randomization - small xyz randomization for robustness
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_Z_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.0

    # Action scaling: (position, rotation, gripper)
    # Lower values = more precise movements
    ACTION_SCALE = np.array([0.1, 0.3, 1])

    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 200
    GRIPPER_SLEEP = 0.5  # Sleep only on state change

    # Compliance parameters for normal operation
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.005,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.002,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005,  # Reduced by 60% (was 0.005)
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
    # Observation keys - cameras used for policy training (wrist-only)
    image_keys = ["wrist"]
    classifier_keys = ["wrist"]
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
        from franka_env.envs.wrappers import SuccessKeyWrapper

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

        # Manual rewards - press 's' anytime during episode to mark success
        env = SuccessKeyWrapper(env)

        env = GripperPenaltyWrapper(env, penalty=-0.005)
        return env
