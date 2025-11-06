
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



#1 - створюємо функцію для генерації простих фігур
def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)
    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 160]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

#2 формуємо набори данних

X = [ ] #список ознак
y = [ ] #список міток


colors = {
    "red": (0, 0, 255), #bgr
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}


for color_name, bgr in colors.items():
    for _ in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(color_name)


#3 - розділяємо данні (70 на 30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#4 - навчаємо саму модель
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)    #будуємо модель

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>1000:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]

            mean_color = cv2.mean(roi)[:3]
            mean_color = np.array(mean_color).reshape(1, -1)


            label = model.predict(mean_color)[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, label.upper(), (x, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("color", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


#
# #5 - перевіряємо точність
# accuracy = model.score(X_test, y_test)
# print(f'Точність моделі: {round(accuracy*100, 2)}%')
#
# test_image = generate_image((0, 250, 0), "circle")
# mean_color = cv2.mean(test_image)[:3]
# prediction = model.predict([mean_color])
# print(f'Передбачення: {prediction[0]}')
#
# cv2.imshow("image", test_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()