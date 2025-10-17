import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.ram_insertion.wrapper import RAMEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://127.0.0.1:5000/" # "http://127.0.0.2:5000/"
    # REALSENSE_CAMERAS = {
    #     "wrist_1": {
    #         "serial_number": "127122270146",
    #         "dim": (1280, 720),
    #         "exposure": 40000,
    #     },
    #     "wrist_2": {
    #         "serial_number": "127122270350",
    #         "dim": (1280, 720),
    #         "exposure": 40000,
    #     },
    # }
    REALSENSE_CAMERAS = {
        "front": {
            "camera_type": "rs", # Realsense
            "serial_number": "215122255213",
            "dim": (1280, 720), # (640, 480),
            "exposure": 40000,
        },
        "wrist": {
            "camera_type": "rs", # Realsense
            "serial_number": "123622270802",
            "dim": (1280, 720), # (640, 480),
            "exposure": 40000,
        }
    }
    # IMAGE_CROP = {
    #     "wrist_1": lambda img: img[150:450, 350:1100],
    #     "wrist_2": lambda img: img[100:500, 400:900],
    # } 
    IMAGE_CROP = {
        "front": lambda img: img[150:450, 350:1100],
        "wrist": lambda img: img[100:500, 400:900],
    }

    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.3, 0.3, 0.15, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.3, 0.3, 0.1, 0.01, 0.1, 0.4])
    
    # TARGET_POSE = np.array([0.5338293080053694,0.016697449706226962,0.083988679136586, np.pi, 0, 0]) # np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    # GRASP_POSE = np.array([0.5955117722053217,-0.29151312450955336,0.09411383106254659, np.pi, 0, 0]) # np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    # RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    # # RESET_POSE = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037]) # TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    # ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.10, 0.09, 0.03, 0.01, 0.1, 0.4])
    # ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.10, 0.09, 0.09, 0.01, 0.1, 0.4])
    # # ABS_POSE_LIMIT_LOW = np.array([0.7006635559910195,-0.38834772336958034,0.030042759985069403,3.1295645034763098,-0.034009936022362686,-0.0043003693062975135]) # TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    # # ABS_POSE_LIMIT_HIGH = np.array([0.05517903015110355,0.4595559103760465,0.7218287550028863,3.0978624840386075,0.06753642592261033,0.014600408652612584]) # TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 0.06, 1)
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 300 # 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["front", "wrist"] # ["wrist_1", "wrist_2"]
    classifier_keys = ["front", "wrist"] # ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
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
                return int(sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
