import argparse
import cv2
import json
import os

# Global variables
base_dir = "."
base_file_name = "init"
file_extension = ".json"
colors = [
    (0, 0, 255),  # Red
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]
inits = ['left', 'middle', 'right']

def get_all_files(base_dir, base_name, extension):
    """Get all files with the specified base name and extension, sorted by index."""
    files = [f for f in os.listdir(base_dir) if f.startswith(base_name) and f.endswith(extension)]
    files.sort(key=lambda x: int(x[len(base_name) + 1:-len(extension)]))
    files = [os.path.join(base_dir, f) for f in files]
    return files

def load_points(file_name):
    """Load points from the specified file."""
    try:
        with open(file_name, "r") as file:
            points = json.load(file)
        print(f"Points loaded from {file_name}: {points}")
        return points
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return []
    
def load_frame(file_name):
    return cv2.imread(file_name)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    files = get_all_files(base_dir, base_file_name, file_extension)
    if not files:
        print("No saved points files found!")
        return

    file_index = 0  # Index of the current file
    points = load_points(files[file_index])

    saved_frames = [f.replace(base_file_name, "frame").replace(file_extension, ".png") for f in files]
    saved_frame = load_frame(saved_frames[file_index])

    success_count = 0
    success_files = []

    cv2.namedWindow("Camera Feed")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured.")
            break

        # Draw the points on the frame
        for idx, (x, y) in enumerate(points):
            color = colors[idx % len(colors)]  # Cycle through colors
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.putText(frame, f"{idx+1}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (255, 0, 0), 5)

        # Overlay saved frame on current frame
        frame = cv2.addWeighted(frame, 0.5, saved_frame, 0.5, 0)
        

        # Display the filename
        cv2.putText(frame, f"File: {files[file_index]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show current success count
        cv2.putText(frame, f"Successes: {success_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show paper towel init position
        # cv2.putText(frame, f"Init: {inits[points[0][0] % 3]}", (10, 90),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit
            break
        elif key == ord("n") or key == 83:  # Next file or Right Arrow
            file_index = (file_index + 1) % len(files)
            points = load_points(files[file_index])
            saved_frame = load_frame(saved_frames[file_index])
        elif key == ord("p") or key == 81:  # Previous file or Left Arrow
            file_index = (file_index - 1 + len(files)) % len(files)
            points = load_points(files[file_index])
            saved_frame = load_frame(saved_frames[file_index])
        elif key == ord("s"):
            success_count += 1
            success_files.append(files[file_index])
            print(f"SUCCESS #{success_count} - File: {files[file_index]}")

            file_index = (file_index + 1) % len(files)
            points = load_points(files[file_index])
            saved_frame = load_frame(saved_frames[file_index])

    cap.release()
    cv2.destroyAllWindows()

    print("summary of successes")
    print(f"total successes: {success_count}")
    print("files that were successful:")
    for f in success_files:
        print("  -", f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="points", help="Base directory to save the points.")
    parser.add_argument("--base_file_name", type=str, default="init", help="Base filename to save the points.")
    args = parser.parse_args()

    base_dir = args.base_dir
    base_file_name = args.base_file_name
    main()
