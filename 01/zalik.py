import cv2
import os
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)

VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
OUT_DIR = os.path.join(PROJECT_DIR, "output")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

VIDEO_PATH = os.path.join(VIDEO_DIR, "zalik.mp4")
cap = cv2.VideoCapture(VIDEO_PATH)
OUTPUT_VIDEO_PATH = os.path.join(OUT_DIR, "zalik.mp4")


CONF_TRESHOLD = 0.4
RESIZE_WIDTH = 1280

VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

model = YOLO('yolov8n.pt')

out = None
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if RESIZE_WIDTH is not None:
        h, w = frame.shape[:2]
        scale = RESIZE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame.shape[1], frame.shape[0]))

    result = model(frame, conf = CONF_TRESHOLD, verbose = False)

    counts = {name: 0 for name in VEHICLE_CLASSES.values()}

    for r in result:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id in VEHICLE_CLASSES:
                class_name = VEHICLE_CLASSES[cls_id]

                counts[class_name] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f'{class_name} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x1, y1 - 10), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

            cv2.rectangle(frame, (0, 0), (200, 160), (0, 0, 0), -1)
            cv2.putText(frame, "Statistik:", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)

            y_offset = 20
            for v_type, count in counts.items():
                y_offset += 25
                color = (0, 255, 0) if count > 0 else (150, 150, 150)
                text = f"{v_type}: {count}"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.1, color, 1)

            if out is not None:
                out.write(frame)

            cv2.imshow("Трафік", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
