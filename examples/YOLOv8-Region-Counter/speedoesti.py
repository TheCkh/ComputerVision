import cv2
from ultralytics import solutions
import csv
import os

# Initialize CSV
csv_path = '/Users/chewk/Downloads/speed_violations.csv'
if os.path.exists(csv_path):
    os.remove(csv_path)

def init_csv(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Video Time (seconds)', 'ID', 'Speed (km/h)', 'Status'])

init_csv(csv_path)

# Setup video
cap = cv2.VideoCapture("/Users/chewk/Downloads/busstop.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize SpeedEstimator with more parameters
speed = solutions.SpeedEstimator(
    show=True,
    model="yolo11n.pt",
    region=[(0, 400), (1280, 400)],  # Line for speed detection
    classes=[0],
    spdl_dist_thresh=10  # Distance threshold for speed calculation
)

frame_count = 0
speed_limit = 5.0
logged_ids = set()

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    frame_count += 1
    video_time = frame_count / fps

    out = speed.estimate_speed(im0)

    # Access the built-in dist_data for speeds
    if hasattr(speed, 'dist_data'):
        for track_id, track_speed in speed.dist_data.items():
            if track_id not in logged_ids:
                status = "EXCEEDED!" if track_speed > speed_limit else "Normal"

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"{video_time:.2f}",
                        track_id,
                        f"{track_speed:.2f}",
                        status
                    ])
                logged_ids.add(track_id)

    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
