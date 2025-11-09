import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


X = []
y = []

colors = {
    "red": (0, 0, 255), #bgr
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "purple": (128, 0, 128),
    "pink": (203, 192, 255),
    "orange": (0, 165, 255),
    "brown": (40, 70, 100)
}

for color_name, bgr in colors.items():
    for _ in range(30):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr)+noise, 0, 255)
        X.append(sample)
        y.append(color_name)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Точність моделі (8 кольорів): {accuracy*100}%')

color_history = []

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000:
            x, y1, w, h = cv2.boundingRect(cnt)
            roi = frame[y1:y1+h, x:x+w]

            mean_color = cv2.mean(roi)[:3]
            color_history.append(mean_color)

            if len(color_history)>10:
                color_history.pop(0)

                smoothed_color = np.mean(color_history, axis = 0)
                model_input = np.array(smoothed_color).reshape(1, -1)

                probabilities = model.predict_proba(model_input)[0]
                confidence = np.max(probabilities)
                label = model.classes_[np.argmax(probabilities)]

                display_text = f"{label.upper()} ({confidence * 100}%)"

                cv2.rectangle(frame, (x, y1), (x + w, y1 + h), (255, 255, 255), 2)
                cv2.putText(frame, display_text, (x, y1 - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()