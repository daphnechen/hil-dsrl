import copy
import time
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests

from franka_env.envs.franka_env import FrankaEnv


class TestCubeEnv(FrankaEnv):
    """Simple cube reaching and grasping environment for testing pipeline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def go_to_reset(self, joint_reset=False):
        """
        Move to the reset position.
        Add a small z offset before going to reset to avoid collision with object.
        """
        # Use precision mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)

        # Pull up first to avoid hitting cube
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04  # Lift 4cm above reset height
        self.interpolate_move(reset_pose, timeout=1)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Cartesian reset
        if self.randomreset:  # Randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)
        time.sleep(0.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        """Reset environment to initial state."""
        if self.save_video:
            self.save_video_recording()

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}
