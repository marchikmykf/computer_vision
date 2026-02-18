import cv2
import os
import numpy as np
import yt_dlp
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
MODEL_PATH = "yolov8n.pt"

QUEUE_ZONE = np.array([
    [430, 330],
    [1100, 200],
    [1300, 400],
    [480, 600]
], np.int32)

CROWD_DISTANCE_THRESHOLD = 90

def get_stream_url(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f" Помилка: {e}")
        return None

print("Завантаження моделі YOLO...")
model = YOLO(MODEL_PATH)

stream_url = get_stream_url(YOUTUBE_URL)
if not stream_url:
    exit()

cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    cv2.fillPoly(overlay, [QUEUE_ZONE], (150, 150, 150))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [QUEUE_ZONE], isClosed=True, color=(255, 255, 255), thickness=2)

    results = model(frame, conf=0.3, classes=[0], verbose=False)

    people_in_queue = []

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = y2

            if cv2.pointPolygonTest(QUEUE_ZONE, (cx, cy), False) >= 0:
                people_in_queue.append((cx, cy, x1, y1, x2, y2))
                color = (0, 255, 255)
            else:
                color = (255, 0, 0)  # Ті, хто поза зоною

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

    num_people = len(people_in_queue)
    close_pairs_count = 0

    for i in range(num_people):
        for j in range(i + 1, num_people):
            c1 = people_in_queue[i]
            c2 = people_in_queue[j]

            dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

            if dist < CROWD_DISTANCE_THRESHOLD:
                close_pairs_count += 1
                cv2.line(frame, (c1[0], c1[1]), (c2[0], c2[1]), (0, 0, 255), 2)

    if num_people == 0:
        queue_status = "EMPTY"
        status_color = (200, 200, 200)
    elif num_people <= 2:
        queue_status = "SHORT"
        status_color = (0, 255, 0)
    elif num_people >= 4 and close_pairs_count >= 2:
        queue_status = "LONG (CROWDED)"
        status_color = (0, 0, 255)
    else:
        queue_status = "MEDIUM"
        status_color = (0, 255, 255)

    cv2.rectangle(frame, (10, 10), (350, 160), (0, 0, 0), -1)
    cv2.putText(frame, "QUEUE ANALYSIS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"People waiting: {num_people}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(frame, f"Close contacts: {close_pairs_count}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Status: {queue_status}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow('Queue Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()