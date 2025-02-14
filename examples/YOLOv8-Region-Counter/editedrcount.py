import argparse
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

# Track history for both position and timing
track_history = defaultdict(list)
track_times = defaultdict(list)

current_region = None
counting_regions = [
    {
        "name": "Speed Zone 1",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        "counts": 0,
        "speeds": [],  # List to store speeds of objects in this region
        "dragging": False,
        "region_color": (255, 42, 4),
        "text_color": (255, 255, 255),
    },
    {
        "name": "Speed Zone 2",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        "counts": 0,
        "speeds": [],
        "dragging": False,
        "region_color": (37, 255, 225),
        "text_color": (0, 0, 0),
    },
]

def calculate_speed(positions, times, fps, pixels_per_meter=30):
    """
    Calculate speed from position history and timestamps.
    
    Args:
        positions: List of (x, y) positions
        times: List of timestamps
        fps: Video frame rate
        pixels_per_meter: Conversion factor from pixels to meters
    
    Returns:
        speed in meters per second
    """
    if len(positions) < 2 or len(times) < 2:
        return 0
    
    # Calculate distance in pixels
    dx = positions[-1][0] - positions[-2][0]
    dy = positions[-1][1] - positions[-2][1]
    distance_pixels = np.sqrt(dx**2 + dy**2)
    
    # Convert to meters
    distance_meters = distance_pixels / pixels_per_meter
    
    # Calculate time difference in seconds
    time_diff = (times[-1] - times[-2]) / fps
    
    if time_diff == 0:
        return 0
        
    speed = distance_meters / time_diff
    return speed

def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for region manipulation."""
    global current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False

def run(
    weights="yolov8n.pt",
    source=None,
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    """Run Region counting and speed estimation on a video using YOLOv8."""
    vid_frame_count = 0
    fps = 30  # Default FPS, will be updated from video

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")
    names = model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width = int(videocapture.get(3))
    frame_height = int(videocapture.get(4))
    fps = int(videocapture.get(5))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.avi"), fourcc, fps, (frame_width, frame_height))

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if classes is None or cls in classes:  # Filter for specified classes
                    annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                    # Update tracking history
                    track = track_history[track_id]
                    track.append((float(bbox_center[0]), float(bbox_center[1])))
                    track_times[track_id].append(vid_frame_count)
                    
                    if len(track) > 30:
                        track.pop(0)
                        track_times[track_id].pop(0)
                    
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                    # Calculate speed and check regions
                    current_speed = calculate_speed(track, track_times[track_id], fps)
                    
                    for region in counting_regions:
                        if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                            region["counts"] += 1
                            region["speeds"].append(current_speed)

        # Draw regions with counts and average speeds
        for region in counting_regions:
            avg_speed = np.mean(region["speeds"]) if region["speeds"] else 0
            region_label = f"Count: {region['counts']} Speed: {avg_speed:.1f} m/s"
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coordinates = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            
            # Draw text and region outline
            cv2.putText(
                frame, region_label, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(
                frame, [polygon_coordinates], 
                isClosed=True, color=region_color, thickness=region_thickness
            )

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter with Speed")
                cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter with Speed", mouse_callback)
            cv2.imshow("Ultralytics YOLOv8 Region Counter with Speed", frame)

        if save_img:
            video_writer.write(frame)

        # Reset counts and speeds for next frame
        for region in counting_regions:
            region["counts"] = 0
            region["speeds"] = []

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")
    return parser.parse_args()

def main(options):
    run(**vars(options))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)