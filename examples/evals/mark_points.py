import argparse

import cv2
import json
import os

import cv2
import json
import os

# Global variables
points = []
n_points = 2  # Number of points to mark
base_dir = "."
base_file_name = "init"
file_extension = ".json"

# Define a list of colors for marking points
colors = [
    (0, 0, 255),  # Red
    (255, 0, 0),  # Blue
    (0, 255, 0),  # Green
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]

def get_next_file_index(base_name, extension):
    """Get the next available file index."""
    index = 0
    while os.path.exists(os.path.join(base_dir, f"{base_name}_{index}{extension}")):
        index += 1
    return index

def click_event(event, x, y, flags, param):
    """Callback function to handle mouse clicks."""
    global points
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

def save_points():
    """Save the marked points to a file with an incremented index."""
    index = get_next_file_index(base_file_name, file_extension)
    file_name = os.path.join(base_dir, f"{base_file_name}_{index}{file_extension}")
    with open(file_name, "w") as file:
        json.dump(points, file)
    print(f"Points saved to {file_name}.")

def save_frame(frame):
    """Save the current frame to a file with an incremented index."""
    index = get_next_file_index("frame", ".png")
    file_name = os.path.join(base_dir, f"frame_{index}.png")
    cv2.imwrite(file_name, frame)
    print(f"Frame saved to {file_name}.")

def main():

    global points

    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    cv2.namedWindow("Camera Feed")
    cv2.setMouseCallback("Camera Feed", click_event)

    # Create base directory if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured.")
            break

        # Draw the marked points on the frame
        for idx, (x, y) in enumerate(points):
            color = colors[idx % len(colors)]  # Cycle through colors if more points than colors
            cv2.circle(frame, (x, y), 5, color, -1)  # Colored dot
            cv2.putText(frame, f"{idx+1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Quit the program
            break
        elif key == ord("s"):  # Save points and reset
            if len(points) == n_points:
                save_points()
                save_frame(frame)
                points = []  # Reset points for the next initialization
            else:
                print(f"Please mark all {n_points} points before saving!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark points on the camera feed.")
    parser.add_argument("--n_points", type=int, default=1, help="Number of points to mark.")
    parser.add_argument("--base_dir", type=str, default="points", help="Base directory to save the points.")
    parser.add_argument("--base_file_name", type=str, default="init", help="Base filename to save the points.")
    args = parser.parse_args()

    n_points = args.n_points
    base_dir = args.base_dir
    base_file_name = args.base_file_name
    main()
    
## STEER DEcay - 5k steps - 12/20