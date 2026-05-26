#!/usr/bin/env python3
from absl import app, flags
import os
import re
import pickle
from typing import Dict, List

import cv2
import imageio
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "buffer_dir",
    None,
    "Path to a replay buffer directory containing transitions_*.pkl files.",
)
flags.DEFINE_string(
    "output_path",
    None,
    "Output video path (e.g., /tmp/replay_front.mp4).",
)
flags.DEFINE_string(
    "camera_key",
    "side",
    "Observation camera key to use when not composing side-by-side.",
)
flags.DEFINE_bool(
    "side_by_side",
    False,
    "If True, compose front-facing and wrist cameras side by side.",
)
flags.DEFINE_string(
    "front_camera_key",
    "side",
    "Front-facing camera key when side_by_side is True.",
)
flags.DEFINE_string(
    "wrist_camera_key",
    "wrist",
    "Wrist camera key when side_by_side is True.",
)
flags.DEFINE_integer(
    "max_step",
    5000,
    "Maximum environment step (timestep) to include in the video.",
)
flags.DEFINE_integer(
    "fps",
    10,
    "Frames per second for the output video.",
)
flags.DEFINE_integer(
    "border_px",
    4,
    "Border thickness in pixels when interventions are active.",
)


def _list_transition_files(buffer_dir: str) -> List[str]:
    if not os.path.isdir(buffer_dir):
        raise ValueError(f"buffer_dir is not a directory: {buffer_dir}")
    files = []
    for fname in os.listdir(buffer_dir):
        if re.match(r"^transitions_\d+\.pkl$", fname):
            files.append(os.path.join(buffer_dir, fname))
    if not files:
        raise ValueError(f"No transitions_*.pkl files found in: {buffer_dir}")
    # Sort by step number in filename.
    files.sort(key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    return files


def _load_transitions(buffer_dir: str) -> List[Dict]:
    transitions: List[Dict] = []
    for fp in _list_transition_files(buffer_dir):
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, list) or not obj:
            continue
        transitions.extend(obj)
    if not transitions:
        raise ValueError(f"No transitions loaded from: {buffer_dir}")
    return transitions


def _extract_frame(tr: Dict, camera_key: str) -> np.ndarray:
    obs = tr.get("next_observations", tr.get("observations"))
    if obs is None or camera_key not in obs:
        available = sorted(list(obs.keys())) if isinstance(obs, dict) else []
        raise KeyError(
            f"camera_key '{camera_key}' not found. Available keys: {available}"
        )
    frame = obs[camera_key]
    # Expected shape: (1, H, W, 3)
    if isinstance(frame, np.ndarray) and frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Unexpected frame shape for key '{camera_key}': {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _compose_side_by_side(
    tr: Dict,
    front_key: str,
    wrist_key: str,
) -> np.ndarray:
    left = _extract_frame(tr, front_key)
    right = _extract_frame(tr, wrist_key)
    # Match heights for clean concatenation.
    if left.shape[0] != right.shape[0]:
        target_h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (int(left.shape[1] * target_h / left.shape[0]), target_h))
        right = cv2.resize(right, (int(right.shape[1] * target_h / right.shape[0]), target_h))
    return np.concatenate([left, right], axis=1)


def _add_intervention_border(frame: np.ndarray, border_px: int) -> np.ndarray:
    # Red border in RGB (frames are stored as RGB in the buffers).
    framed = frame.copy()
    h, w = framed.shape[:2]
    b = max(int(border_px), 1)
    red = np.array([255, 0, 0], dtype=framed.dtype)
    framed[:b, :, :] = red
    framed[-b:, :, :] = red
    framed[:, :b, :] = red
    framed[:, -b:, :] = red
    return framed


def main(_):
    if not FLAGS.buffer_dir or not FLAGS.output_path:
        raise ValueError("--buffer_dir and --output_path are required.")

    transitions = _load_transitions(FLAGS.buffer_dir)

    # Sort by timestep if present.
    if "timestep" in transitions[0]:
        transitions = sorted(
            transitions, key=lambda tr: tr.get("timestep", 0)
        )

    frames = []
    for tr in transitions:
        step = tr.get("timestep", None)
        if step is not None and step > FLAGS.max_step:
            break
        if FLAGS.side_by_side:
            frame = _compose_side_by_side(
                tr,
                FLAGS.front_camera_key,
                FLAGS.wrist_camera_key,
            )
        else:
            frame = _extract_frame(tr, FLAGS.camera_key)
        intervene = "info" in tr and "intervene_action" in tr["info"]
        if intervene:
            frame = _add_intervention_border(frame, FLAGS.border_px)
        frames.append(frame)

    if not frames:
        raise ValueError("No frames generated. Check camera_key or max_step.")

    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.output_path)), exist_ok=True)
    imageio.mimsave(
        FLAGS.output_path,
        frames,
        fps=FLAGS.fps,
        macro_block_size=None,
    )
    print(f"Saved {len(frames)} frames to {FLAGS.output_path}")


if __name__ == "__main__":
    app.run(main)
