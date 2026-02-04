
import cv2
import os
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, "video")
OUT_DIR = os.path.join(PROJECT_DIR, "output")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

USE_WEBCAM = False

if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    VIDEO_PATH = os.path.join(VIDEO_DIR, "video_third.mp4")
    cap = cv2.VideoCapture(VIDEO_PATH)

model = YOLO('yolov8n.pt')

CONF_THRESHOLD = 0.4
RESIZE_WIDTH = 960

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16

prev_time = time.time()

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

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    cats_count = 0
    dogs_count = 0

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAT_CLASS_ID:
                cats_count += 1
                color = (255, 0, 0)
                label_name = "Cat"
            elif cls == DOG_CLASS_ID:
                dogs_count += 1
                color = (0, 0, 255)
                label_name = "Dog"
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f'{label_name} {conf:.2f}'
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    total_animals = cats_count + dogs_count

    now = time.time()
    dt = now - prev_time
    prev_time = now
    fps = 1.0 / dt if dt > 0 else 0

    cv2.rectangle(frame, (10, 10), (250, 130), (0, 0, 0), -1)

    cv2.putText(frame, f'Cats: {cats_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f'Dogs: {dogs_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f'Total: {total_animals}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Cats & Dogs", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()