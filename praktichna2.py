import cv2
import numpy as np

img = cv2.imread('images/KR.jpg')
img = cv2.resize(img, (img.shape[1]//5, img.shape[0]//5))
img_copy = img.copy()
img = cv2.GaussianBlur(img, (3,3), 5)

xy = []

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 29, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img, lower, upper)

img = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) > 8:
            shape = "Oval"
        else:
            shape = "other"

        x, y, w, h = cv2.boundingRect(cnt)
        xy.append([x, y])

        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(img_copy, f'A: {area}, ({x}, {y})', (x, y-5), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0 ,0), 1)
        cv2.putText(img_copy, f'Shape: {shape}', (x, y - 25), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0 ,0), 1)

cv2.putText(img_copy, f'Color: green', (xy[0][0], xy[0][1]-15), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0 ,0), 1)
cv2.putText(img_copy, f'Color: yellow', (xy[1][0], xy[1][1]-15), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0 ,0), 1)
cv2.putText(img_copy, f'Color: red', (xy[2][0], xy[2][1]-15), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0 ,0), 1)
cv2.putText(img_copy, f'Color: blue', (xy[3][0], xy[3][1]-15), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0 ,0), 1)

#print(xy)

cv2.imshow('orig', img)
cv2.imshow('mask', img_copy)
cv2.imwrite('result.jpg', img_copy)
cv2.waitKey(0)

cv2.destroyAllWindows()
