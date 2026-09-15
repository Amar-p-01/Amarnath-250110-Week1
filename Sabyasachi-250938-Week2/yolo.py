from pathlib import Path
import cv2
from ultralytics import YOLO
import time

VIDEO_NAME = "yolo_input.mp4"
OUTPUT_NAME = "output_detected.mp4"
CONF_THRESHOLD = 0.35

script_dir = Path(__file__).resolve().parent
input_path = script_dir / VIDEO_NAME
output_path = script_dir / OUTPUT_NAME


model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(str(input_path))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

print("Making predictions")
t = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, classes=[0], conf=CONF_THRESHOLD, persist=False, verbose=False)

    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"#{track_id} {conf:.2f}" if track_id is not None else f"{conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    out.write(frame)

print(f"{time.time()-t}s To decode video of duration {cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)} s")
cap.release()
out.release()
print(f"Finished. Saved to {output_path}")