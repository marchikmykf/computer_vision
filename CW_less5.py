import cv2
import numpy as np

img = cv2.imread('images/cat.jpg')
img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
img_copy = img.copy()
img = cv2.GaussianBlur(img, (3, 3), 5)

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 16, 97])  #взяли мінімальний поріг зображення
upper = np.array([27, 95, 213]) #взяли максимальний поріг зображення
mask = cv2.inRange(img, lower, upper)

img = cv2.bitwise_and(img, img, mask=mask)   #накладаємо маску на наше зображення
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        perimeter = cv2.arcLength(cnt, True)#closed
        M = cv2.moments(cnt)  #момент контуру

        #центр мас
        if M["m00"] !=0: #якщо контур замкнений
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])  #середня позція контура

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2) #відношення ширини до висоти
        #міра округлості об`єкта
        compactness = round((4 * np.pi * area) / (perimeter**2), 2)

        #дізнаємося форму об'єкта
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) >8:
            shape = "oval"
        else:
            shape = "inshe"

        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (0, 255, 0),-1)
        cv2.putText(img_copy, f'shape:{shape}', (x, y-25), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.putText(img_copy, f'A:{int(area)}, P:{int(perimeter)}', (x, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, f'AR:{aspect_ratio}, C:{compactness}', (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)


cv2.imshow('Original', img)
cv2.imshow('mask', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()