import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    LastNGripperActionsWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.furniture_bench.wrapper import FurnitureBenchEnv
from experiments.usb_pickup_insertion.wrapper import GripperPenaltyWrapper

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        # "front": {
        #     "camera_type": "rs", # Realsense
        #     "serial_number": "215122255213",
        #     "dim": (1280, 720), # (640, 480),
        #     "exposure": 40000,
        # },
        "side": {
            "camera_type": "rs", # Realsense
            "serial_number": "947122060531",
            "dim": (1280, 720), # (640, 480),
            "exposure": 40000,
        },
        "wrist": {
            "camera_type": "rs", # Realsense
            "serial_number": "123622270802",
            "dim": (640, 480), # (848, 480)
            "exposure": 40000,
        }
    }

    IMAGE_CROP = {
        # "front": lambda img: img[180:430, 150:550],  
        "side": lambda img: img[70:470, 450:1150], 
    }

    # Step 1: add reset pose
    RESET_POSE = np.array([0.48, 0.04, 0.24, np.pi, 0, np.pi / 2])

    # Step 2: add bounding boxes
    # ABS_POSE_LIMIT_LOW  = np.array([0.42, -0.14, 0.20, np.pi - 0.05, -0.05, np.pi / 2 - 0.05])
    # ABS_POSE_LIMIT_HIGH = np.array([0.51,  0.24, 0.29, np.pi + 0.05,  0.05, np.pi / 2 + 0.05])
    ABS_POSE_LIMIT_LOW  = np.array([0.42, -0.44, 0.10, np.pi - 0.05, -0.05, np.pi / 2 - 0.05])
    ABS_POSE_LIMIT_HIGH = np.array([0.61,  0.44, 0.29, np.pi + 0.05,  0.05, np.pi / 2 + 0.05])

    # Step 3: set reset randomization
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 6
    ACTION_SCALE = np.array([0.05, 0.3, 1]) # Testing x
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 400


    ## Step 4: Copy the same control parameters -- keep it the same as below : TODO: figure out how to tune these+
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
    image_keys = ["side", "wrist"] #["front", "side", "wrist"] # ["side_1"]
    classifier_keys = ["wrist", "side"] # ["front", "side", "wrist"] # ["side_1"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    # buffer_period = 1000
    # checkpoint_period = 5000
    # steps_per_update = 50
    pretraining_steps = 0 # How many steps to pre-train the model for using RLPD on offline data only.
    reward_scale = 1 # How much to scale actual rewards (not RLIF penalties) for RLIF.
    rlif_minus_one = False
    checkpoint_period = 4000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    batch_size = 64
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env_config = EnvConfig()
        assert np.logical_and(env_config.ABS_POSE_LIMIT_LOW <= env_config.RESET_POSE, env_config.RESET_POSE <= env_config.ABS_POSE_LIMIT_HIGH).all()
        env = FurnitureBenchEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        # env = LastNGripperActionsWrapper(env, 8)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # added check for z position to further robustify classifier, but should work without as well
                prob = sigmoid(classifier(obs))[0]
                if prob > 0.1:
                    print(f"{prob:.2f}")
                return int(prob > 0.90) # and obs['state'][0, 3] < 0.12
                # return 0.0

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env
