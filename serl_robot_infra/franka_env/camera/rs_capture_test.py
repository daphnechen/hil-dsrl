import pyrealsense2 as rs
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

frames_count = 0
t0 = time.time()

try:
    while frames_count < 60:  # ~10 seconds at 30 fps
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frames_count += 1
finally:
    pipeline.stop()

elapsed = time.time() - t0
print("Processed frames:", frames_count)
print("Loop FPS:", frames_count / elapsed)
