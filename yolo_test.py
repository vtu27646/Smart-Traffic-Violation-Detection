from ultralytics import YOLO
import cv2
import os
from datetime import datetime

model = YOLO("yolo11n.pt")

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("ERROR: Could not open the video.")
    exit()

os.makedirs("evidence", exist_ok=True)

print("Improved stop-line detection started!")
print("Press Q to stop.")

previous_positions = {}
violated_ids = set()

while True:
    success, frame = video.read()

    if not success:
        print("End of video.")
        break

    results = model.track(
        frame,
        classes=[1, 2, 3, 5, 7],
        persist=True,
        verbose=False
    )

    stop_line_y = 1350
    annotated_frame = results[0].plot()

    # Draw stop line
    cv2.line(
        annotated_frame,
        (0, stop_line_y),
        (1080, stop_line_y),
        (0, 0, 255),
        5
    )

    cv2.putText(
        annotated_frame,
        "STOP LINE",
        (20, stop_line_y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, vehicle_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if vehicle_id in previous_positions:

                previous_y = previous_positions[vehicle_id]

                if (
                    previous_y < stop_line_y
                    and center_y >= stop_line_y
                    and vehicle_id not in violated_ids
                ):

                    violated_ids.add(vehicle_id)

                    timestamp = datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )

                    filename = (
                        f"evidence/"
                        f"stopline_{timestamp}"
                        f"_vehicle_{vehicle_id}.jpg"
                    )

                    # Highlight violating vehicle
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        6
                    )

                    # Violation label near vehicle
                    label_y = max(y1 - 20, 40)

                    cv2.putText(
                        annotated_frame,
                        "STOP-LINE VIOLATION",
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )

                    # Timestamp
                    cv2.putText(
                        annotated_frame,
                        timestamp,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2
                    )

                    # Save evidence
                    cv2.imwrite(
                        filename,
                        annotated_frame
                    )

                    print("Violation detected!")
                    print("Vehicle ID:", vehicle_id)
                    print("Evidence saved:", filename)

            previous_positions[vehicle_id] = center_y

    display_frame = cv2.resize(
        annotated_frame,
        (450, 800)
    )

    cv2.imshow(
        "Stop-Line Violation Detection",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()