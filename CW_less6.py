import cv2
import numpy as np

cap = cv2.VideoCapture(0) #0 - WEBCAMERA
ret, frame1 = cap.read()
grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grey1 = cv2.convertScaleAbs(grey1, alpha=1.5, beta=15)  # контрастність

while True:
    ret, frame2 = cap.read()         #frame - кадр який перебирається ret -чи відкритий кадр
    if not ret: #Якщо не отпимали кадри
        print("Відео скінчилося")
        break

    grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    grey2 = cv2.convertScaleAbs(grey2, alpha=1.5, beta=15)

    diff = cv2.absdiff(grey1, grey2) #difference between
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1700:
            x, y, h, w = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", frame2)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()