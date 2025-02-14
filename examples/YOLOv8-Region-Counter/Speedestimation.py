import cv2
from ultralytics import solutions
import csv
import os

# First check if file exists and remove it
if os.path.exists('/Users/chewk/Downloads/speed_violations.csv'):
    os.remove('/Users/chewk/Downloads/speed_violations.csv')

def init_csv(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Video Time (seconds)', 'ID', 'Raw Speed', 'Speed (km/h)', 'Status', 'Class'])

# Calibration factors
PERSON_HEIGHT_PIXELS = 170
PERSON_HEIGHT_METERS = 1.7
PIXEL_TO_METER = PERSON_HEIGHT_METERS / PERSON_HEIGHT_PIXELS

# Define a straight line
#speed_line = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

speed_line = [(0, 400), (1280, 400)]

csv_path = '/Users/chewk/Downloads/speed_violations.csv'
init_csv(csv_path)
speed_limit = 5.0  # km/h
logged_ids = set()

cap = cv2.VideoCapture("/Users/chewk/Downloads/busstop.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("speed_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

speed = solutions.SpeedEstimator(
    show=True,
    model="yolo11n.pt",
    region=speed_line,
    classes=[0] #class 0 - recognises only people
)

frame_count = 0

def get_video_time(frame_count, fps):
    return frame_count / fps

def calibrate_speed(raw_speed):
    """Convert raw pixel speed to real-world speed in km/h"""
    speed_ms = raw_speed * PIXEL_TO_METER
    speed_kmh = speed_ms * 3.6
    return speed_kmh

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    frame_count += 1
    video_time = get_video_time(frame_count, fps)

    out = speed.estimate_speed(im0)

    if hasattr(speed, 'spd'):
        for track_id, track_speed in speed.spd.items():
            if track_id not in logged_ids:
                raw_speed = float(track_speed)
                calibrated_speed = calibrate_speed(raw_speed)

                # Determine if speed limit exceeded
                status = "EXCEEDED!" if calibrated_speed > speed_limit else "Normal"

                print(f"\nTrack ID: {track_id}")
                print(f"Raw Speed: {raw_speed}")
                print(f"Speed: {calibrated_speed:.2f} km/h")
                print(f"Status: {status}")

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"{video_time:.2f}",
                        track_id,
                        f"{raw_speed:.2f}",
                        f"{calibrated_speed:.2f}",
                        status,
                        'person'
                    ])
                logged_ids.add(track_id)

    video_writer.write(im0)

print(f"Total unique IDs logged: {len(logged_ids)}")
cap.release()
video_writer.release()
cv2.destroyAllWindows()
