import cv2
import os
import csv
import yt_dlp
from ultralytics import YOLO

# --- НАЛАШТУВАННЯ ЛІНІЙ ---

# Лінія СТАРТУ (Синя)
LINE_START = [(50, 650), (1250, 800)]

# Лінія ФІНІШУ (Червона)
LINE_END = [(700, 320), (1600, 430)]

DISTANCE_METERS = 20
FRAME_SKIP = 2  # Обробка кожного 2-го кадру

# --- ПАПКИ ТА ФАЙЛИ ---
PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_DIR = os.path.join(OUTPUT_DIR, 'videos')
os.makedirs(VIDEO_DIR, exist_ok=True)

OUTPUT_VIDEO_PATH = os.path.join(VIDEO_DIR, 'final_fast_output.mp4')
CSV_PATH = os.path.join(OUTPUT_DIR, 'car_statistics.csv')

# --- ПАРАМЕТРИ ---
YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
MODEL_PATH = "yolov8n.pt"
TRACKER = "bytetrack.yaml"
SAVE_VIDEO = True


# --- ФУНКЦІЯ ОТРИМАННЯ ПОСИЛАННЯ ---
def get_stream_url(url):
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    print(f"📡 З'єднання з YouTube через yt-dlp...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f" Помилка: {e}")
        return None


# --- ЗАПУСК ---
print("Завантаження моделі...")
model = YOLO(MODEL_PATH)

stream_url = get_stream_url(YOUTUBE_URL)
if not stream_url:
    exit()

cap = cv2.VideoCapture(stream_url)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30

writer = None
if SAVE_VIDEO:
    save_fps = fps / FRAME_SKIP
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, save_fps, (frame_width, frame_height))

with open(CSV_PATH, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Class', 'ID', 'Speed (km/h)', 'Timestamp'])

tracker_data = {}  # Тепер зберігає словник: {'frame': номер_кадру, 'line': 'start' або 'end'}
previous_positions = {}
car_speeds = {}
unique_ids = set()


def intersect(A, B, C, D):
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Логіка пришвидшення
    if frame_count % FRAME_SKIP != 0:
        continue

    results = model.track(frame, conf=0.5, tracker=TRACKER, persist=True, verbose=False, classes=[2, 3, 5, 7])

    # Малюємо лінії
    cv2.line(frame, LINE_START[0], LINE_START[1], (255, 0, 0), 2)  # Синя
    cv2.line(frame, LINE_END[0], LINE_END[1], (0, 0, 255), 2)  # Червона

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()

        unique_ids.update(ids.astype(int))

        for box, track_id, cls_id in zip(boxes, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            class_name = model.names[int(cls_id)]

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            curr_pos = (cx, cy)

            if tid in previous_positions:
                prev_pos = previous_positions[tid]

                # --- ЛОГІКА ДВОСТОРОННЬОГО РУХУ ---

                # Перевірка перетину СИНЬОЇ лінії (START)
                if intersect(prev_pos, curr_pos, LINE_START[0], LINE_START[1]):
                    # Сценарій: Машина їхала з Червоної -> на Синю (вже є в базі з поміткою 'end')
                    if tid in tracker_data and tracker_data[tid]['line'] == 'end':
                        start_frame = tracker_data[tid]['frame']
                        end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                        frames_diff = end_frame - start_frame
                        # abs() потрібен, бо різниця може бути від'ємною, якщо ми помилились з порядком,
                        # але тут ми рухаємось вперед по часу, тож просто для безпеки > 1
                        if frames_diff > 1:
                            time_sec = frames_diff / fps
                            speed_kmh = (DISTANCE_METERS / time_sec) * 3.6
                            int_speed = int(speed_kmh)
                            car_speeds[tid] = int_speed
                            print(f"{class_name} {tid} (Red->Blue): {int_speed} км/год")

                            with open(CSV_PATH, mode='a', newline='') as f:
                                writer_csv = csv.writer(f)
                                writer_csv.writerow([class_name, tid, int_speed, f"{time_sec:.2f}"])
                            del tracker_data[tid]

                    # Сценарій: Машина тільки заїхала на Синю лінію (початок шляху Blue->Red)
                    elif tid not in tracker_data:
                        tracker_data[tid] = {'frame': cap.get(cv2.CAP_PROP_POS_FRAMES), 'line': 'start'}

                # Перевірка перетину ЧЕРВОНОЇ лінії (END)
                if intersect(prev_pos, curr_pos, LINE_END[0], LINE_END[1]):
                    # Сценарій: Машина їхала з Синьої -> на Червону (вже є в базі з поміткою 'start')
                    if tid in tracker_data and tracker_data[tid]['line'] == 'start':
                        start_frame = tracker_data[tid]['frame']
                        end_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

                        frames_diff = end_frame - start_frame
                        if frames_diff > 1:
                            time_sec = frames_diff / fps
                            speed_kmh = (DISTANCE_METERS / time_sec) * 3.6
                            int_speed = int(speed_kmh)
                            car_speeds[tid] = int_speed
                            print(f" {class_name} {tid} (Blue->Red): {int_speed} км/год")

                            with open(CSV_PATH, mode='a', newline='') as f:
                                writer_csv = csv.writer(f)
                                writer_csv.writerow([class_name, tid, int_speed, f"{time_sec:.2f}"])
                            del tracker_data[tid]

                    # Сценарій: Машина тільки заїхала на Червону лінію (початок шляху Red->Blue)
                    elif tid not in tracker_data:
                        tracker_data[tid] = {'frame': cap.get(cv2.CAP_PROP_POS_FRAMES), 'line': 'end'}

            previous_positions[tid] = curr_pos

            # Візуалізація
            label = f"{class_name}-{tid}"
            if tid in car_speeds:
                label += f" {car_speeds[tid]}km/h"
                color_box = (0, 0, 255)
            else:
                color_box = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color_box, 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + t_size[0], y1), color_box, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Статистика
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 90), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    count_text = f"Total Vehicles: {len(unique_ids)}"
    speed_text = f"Speed Calculated: {len(car_speeds)}"

    cv2.putText(frame, "STATISTICS:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, count_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, speed_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    cv2.imshow('Speed Camera (Two-Way)', frame)
    if writer:
        writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()