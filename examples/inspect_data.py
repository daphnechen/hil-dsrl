from absl import app, flags
import time
import numpy as np
import os
import pickle
import imageio
import cv2
import matplotlib.cm as cm

FLAGS = flags.FLAGS


def motion_distribution_overlay(
    images,
    threshold_factor: float = 2.0,
    colormap: str = "jet",
    alpha: float = 0.6,
):
    """
    Combine a stack of mostly-identical images with a moving object into a single
    visualization showing the distribution of the object's locations.

    Parameters
    ----------
    images : list[np.ndarray] or np.ndarray
        List or array of images with shape (H, W, 3) or a stacked array
        of shape (N, H, W, 3). All images must have the same shape and dtype.
        Typically uint8 in [0, 255] or float in [0, 1].
    threshold_factor : float, optional
        Controls how aggressively we detect "motion". Larger => only strong
        deviations from background are counted. Default is 2.0.
    colormap : str, optional
        Matplotlib colormap name used for the heatmap ("jet", "viridis", etc.).
    alpha : float, optional
        Opacity of the heatmap when blending with the background (0–1).

    Returns
    -------
    overlay_uint8 : np.ndarray
        RGB image of shape (H, W, 3), dtype uint8, where the background is the
        median scene and the moving object's distribution is shown as a heatmap.
    """

    # Stack into array of shape (N, H, W, 3)
    if isinstance(images, list):
        imgs = np.stack(images, axis=0)
    else:
        imgs = images
    assert imgs.ndim == 4, "Expected images of shape (N, H, W, 3)"

    N, H, W, C = imgs.shape
    assert C == 3, "Expected 3-channel (RGB) images"

    # Convert to float32 in [0, 1] for safe processing
    if imgs.dtype == np.uint8:
        imgs_f = imgs.astype(np.float32) / 255.0
    else:
        imgs_f = imgs.astype(np.float32)

    # 1) Estimate static background as per-pixel median across time
    background = np.median(imgs_f, axis=0)  # (H, W, 3)

    # 2) Measure per-pixel deviation from background for each frame
    #    Using L2 norm over channels; shape (N, H, W)
    diffs = np.linalg.norm(imgs_f - background[None, ...], axis=-1)

    # 3) Turn deviations into binary "object present" masks per frame
    #    Global threshold based on mean deviation across all pixels/frames.
    global_mean_diff = diffs.mean()
    thresh = threshold_factor * global_mean_diff
    masks = diffs > thresh  # (N, H, W), boolean

    # 4) Heatmap: how frequently each pixel is part of the moving object
    #    value in [0, 1] = fraction of frames where that pixel was "active"
    heat = masks.mean(axis=0)  # (H, W)

    # Normalize heat to [0, 1] (avoid division by zero)
    max_heat = heat.max()
    if max_heat > 0:
        heat_norm = heat / max_heat
    else:
        heat_norm = heat

    # 5) Map heat to RGB using a colormap (drop alpha channel)
    cmap = cm.get_cmap(colormap)
    heat_rgb = cmap(heat_norm)[..., :3]  # (H, W, 3)

    # 6) Blend heatmap with background
    overlay = (1.0 - alpha) * background + alpha * heat_rgb

    # 7) Convert back to uint8 in [0, 255]
    overlay_uint8 = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return overlay_uint8


def _get_folder():
    print()
    print("Please provide a folder that's either a /buffer/, /demo_buffer/, /preference_buffer/, or /interventions/.")
    folder = input(" ")
    print(folder)
    return folder


def _save_video(frames: np.ndarray, default_path: str, speed: int = 1):
    print()
    print("Output path: ")
    output_path = input(f"[{default_path}] ")
    if output_path == "":
        output_path = f"{default_path}"
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print()
        yn = input("Will override previous video. Proceed? [y/n] [n] ")
        if yn != 'y':
            print("Exited.")
            return
    print(output_path)
    print()
    imageio.mimsave(
        uri=output_path,
        ims=frames,
        fps=10 * speed,
        macro_block_size=None  # This can help with sizes not multiples of 16
    )


def inspect_buffer(path: str):
    if os.path.isfile(path):
        files = [path]
    else:
        files = [os.path.join(path, fp) for fp in os.listdir(path) if os.path.isfile(os.path.join(path, fp))]
    transitions = []
    for fp in files:
        assert os.path.basename(fp).endswith(".pkl")
        if not (os.path.basename(fp).endswith(".pkl")):
            continue
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, list)
        assert len(obj) > 0
        assert isinstance(obj[0], dict)
        assert set(obj[0].keys()).issuperset(["observations", "actions", "next_observations"])
        transitions += obj
    
    done_indices = [i for i, o in enumerate(transitions) if o['dones'] == 1]
    print()
    print("Pick a trajectory from the following end timesteps:")
    print(done_indices + ["all"])
    done_index = input(f"[{done_indices[0]}] ")
    if done_index == "":
        done_index = done_indices[0]
    if done_index == "all":
        done_index = len(transitions) - 1
        start_index = 0
    else:
        done_index = int(done_index)
        assert done_index in done_indices
        start_index = done_indices[done_indices.index(done_index) - 1] + 1 if done_indices.index(done_index) - 1 >= 0 else 0
    print(done_index)

    camera_keys = sorted(list(transitions[start_index]["observations"].keys() - {"state"})) + ['all']
    assert len(camera_keys) > 0
    print()
    print("Pick a camera from the following cameras:")
    print(camera_keys)
    camera_key = input(f"['{camera_keys[0]}'] ")
    if camera_key == "":
        camera_key = camera_keys[0]
    assert camera_key in camera_keys
    print(camera_key)

    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 1
    FONT_Y = 35
    SPEED = 8

    frames = []
    last_start = 0
    num_interventions = 0
    for i, tr in enumerate(transitions[start_index:done_index+1]):
        if camera_key != 'all':
            left_frame = tr["observations"][camera_key]
            right_frame = tr["next_observations"][camera_key]
        else:
            assert len(camera_keys) == 3
            left_frame = tr["next_observations"][camera_keys[0]]
            right_frame = tr["next_observations"][camera_keys[1]]
        assert isinstance(left_frame, np.ndarray) and left_frame.dtype == np.uint8 and left_frame.shape == (1, 128, 128, 3)
        assert isinstance(right_frame, np.ndarray) and right_frame.dtype == np.uint8 and right_frame.shape == (1, 128, 128, 3)
        left_frame, right_frame = left_frame[0], right_frame[0]
        actions = tr["actions"]
        intervene = "info" in tr and "intervene_action" in tr["info"]
        num_interventions += int(intervene)
        
        new_frame = np.zeros((450, 800, 3), dtype=np.uint8)

        new_frame[50:,:400,:] = cv2.resize(left_frame, (400, 400))
        new_frame[50:,400:,:] = cv2.resize(right_frame, (400, 400))

        new_frame = cv2.putText(
            img = new_frame,
            text = f"t={i}",
            org = (20, FONT_Y),
            fontFace = FONT_FACE,
            fontScale = FONT_SCALE,
            color = (255, 255, 255),
            thickness = FONT_THICKNESS,
            lineType = cv2.LINE_AA,
        )

        # new_frame = cv2.putText(
        #     img = new_frame,
        #     text = f"r={tr['rewards']}",
        #     org = (100, FONT_Y),
        #     fontFace = FONT_FACE,
        #     fontScale = FONT_SCALE,
        #     color = (255, 255, 255),
        #     thickness = FONT_THICKNESS,
        #     lineType = cv2.LINE_AA,
        # )

        if camera_key != 'all':
            new_frame = cv2.putText(
                img = new_frame,
                text = f"act={[f'{a:.3f}' for a in actions]}",
                org = (125, FONT_Y),
                fontFace = FONT_FACE,
                fontScale = 0.6,
                color = (255, 255, 255),
                thickness = FONT_THICKNESS,
                lineType = cv2.LINE_AA,
            )
        else:
            title = "STEER decay"
            title_width = cv2.getTextSize(title, FONT_FACE, FONT_SCALE, FONT_THICKNESS)[0][0]
            new_frame = cv2.putText(
                img = new_frame,
                text = title,
                org = (400 - title_width // 2, FONT_Y),
                fontFace = FONT_FACE,
                fontScale = FONT_SCALE,
                color = (255, 255, 255),
                thickness = FONT_THICKNESS,
                lineType = cv2.LINE_AA,
            )
            new_frame = cv2.putText(
                img = new_frame,
                text = f"#interventions={num_interventions}",
                org = (560, FONT_Y),
                fontFace = FONT_FACE,
                fontScale = FONT_SCALE,
                color = (255, 255, 255),
                thickness = FONT_THICKNESS,
                lineType = cv2.LINE_AA,
            )

        new_frame = cv2.putText(
            img = new_frame,
            text = f"{SPEED}x",
            org = (750, 450 - FONT_Y),
            fontFace = FONT_FACE,
            fontScale = FONT_SCALE,
            color = (255, 255, 255),
            thickness = FONT_THICKNESS * 2,
            lineType = cv2.LINE_AA,
        )

        if intervene:
            cv2.rectangle(new_frame, (2, 52), (400, 450-2), (255, 0, 0), 3)
            cv2.rectangle(new_frame, (400, 52), (800-2, 450-2), (255, 0, 0), 3)

        frames.append(new_frame)
        if tr['dones'] == 1:
            last_start = i + 1
    frames = np.stack(frames, axis=0)

    _save_video(frames, "./media/steer_decay_new.mp4", speed=SPEED)


def inspect_reset_distribution(path: str):
    if os.path.isfile(path):
        files = [path]
    else:
        files = [os.path.join(path, fp) for fp in os.listdir(path) if os.path.isfile(os.path.join(path, fp))]
    transitions = []
    for fp in files:
        assert os.path.basename(fp).endswith(".pkl")
        if not (os.path.basename(fp).endswith(".pkl")):
            continue
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, list)
        assert len(obj) > 0
        assert isinstance(obj[0], dict)
        assert set(obj[0].keys()).issuperset(["observations", "actions", "next_observations"])
        transitions += obj
    
    start_index = 0
    done_index = len(transitions) - 1

    camera_key = 'front'

    FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 1
    FONT_Y = 35
    SPEED = 8

    frames = []
    average_frames = []
    last_start = 0
    num_interventions = 0
    counts = 0
    for i, tr in enumerate(transitions[start_index:done_index+1]):
        left_frame = tr["next_observations"][camera_key]
        assert isinstance(left_frame, np.ndarray) and left_frame.dtype == np.uint8 and left_frame.shape == (1, 128, 128, 3)
        left_frame = left_frame[0]
        if i == last_start:
            frames.append(left_frame)
        
        if len(frames) >= 10 or (i == done_index and len(frames) > 0):
            # average_frame = motion_distribution_overlay(frames)
            average_frame = np.stack(frames, axis=0).mean(axis=0).astype(np.uint8)
            average_frame = cv2.resize(average_frame, (400, 400))
            new_frame = np.zeros((450, 400, 3), dtype=np.uint8)
            new_frame[50:,:,:] = average_frame

            title = "STEER decay"
            title_width = cv2.getTextSize(title, FONT_FACE, FONT_SCALE, FONT_THICKNESS)[0][0]
            new_frame = cv2.putText(
                img = new_frame,
                text = title,
                org = (200 - title_width // 2, FONT_Y),
                fontFace = FONT_FACE,
                fontScale = FONT_SCALE,
                color = (255, 255, 255),
                thickness = FONT_THICKNESS,
                lineType = cv2.LINE_AA,
            )

            average_frames.append(new_frame)
            frames = []

        if tr['dones'] == 1:
            last_start = i + 1

    average_frames = np.stack(average_frames, axis=0)
    _save_video(average_frames, "./media/steer_decay_new.mp4", speed=0.1)


def inspect_classifier_data(path: str):
    assert os.path.exists(path), path
    files = [os.path.join(path, fp) for fp in os.listdir(path) if os.path.isfile(os.path.join(path, fp))]
    successes = []
    failures = []
    for fp in files:
        assert os.path.basename(fp).endswith(".pkl")
        if not os.path.basename(fp).endswith(".pkl"):
            continue
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, list)
        if len(obj) == 0:
            continue
        assert isinstance(obj[0], dict)
        assert set(obj[0].keys()).issuperset(['observations', 'actions', 'next_observations', 'rewards', 'masks', 'dones'])
        if "success" in os.path.basename(fp):
            successes += obj
        elif "failure" in os.path.basename(fp):
            failures += obj
        else:
            raise Exception()
    
    camera_keys = list(successes[0]["observations"].keys() - {"state"})
    assert len(camera_keys) > 0
    print()
    print("Pick a camera from the following cameras:")
    print(camera_keys)
    camera_key = input(f"['{camera_keys[0]}'] ")
    if camera_key == "":
        camera_key = camera_keys[0]
    assert camera_key in camera_keys
    print(camera_key)

    success_frames = []
    for i, tr in enumerate(successes):
        frame = tr["observations"][camera_key]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (1, 128, 128, 3)
        frame = frame[0]
        new_frame = np.zeros((450, 400, 3), dtype=np.uint8)
        new_frame[50:,:,:] = cv2.resize(frame, (400, 400))
        success_frames.append(new_frame)
    success_frames = np.stack(success_frames, axis=0)

    failure_frames = []
    for i, tr in enumerate(failures):
        frame = tr["observations"][camera_key]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (1, 128, 128, 3)
        frame = frame[0]
        new_frame = np.zeros((450, 400, 3), dtype=np.uint8)
        new_frame[50:,:,:] = cv2.resize(frame, (400, 400))
        failure_frames.append(new_frame)
    failure_frames = np.stack(failure_frames, axis=0)

    _save_video(success_frames, "./successes.mp4")
    _save_video(failure_frames, "./failures.mp4")


def main(_):
    folder = _get_folder()
    path = os.path.normpath(folder)
    inspect_reset_distribution(path)
    # folder_name = os.path.basename(path) if os.path.isdir(path) else os.path.basename(os.path.dirname(path))
    # if folder_name == "buffer" or folder_name == "bc_buffer":
    #     inspect_buffer(path)
    # elif folder_name == "classifier_data":
    #     inspect_classifier_data(path)
    # elif folder_name == "demo_data" or folder_name == "evals":
    #     inspect_buffer(path)
    # else:
    #     print(folder_name)
    #     inspect_buffer(path)
    #     raise Exception()

if __name__ == "__main__":
    app.run(main)
