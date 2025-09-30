import numpy as np
import cv2

img = cv2.imread("images/me.jpg")
img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
print(img.shape)
cv2.rectangle(img,(270,90),(400,230),(40, 7, 250),2)
cv2.putText(img, "Lyuta Mariia", (250, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (40, 7, 250), 1)

cv2.imshow("Image", img)
cv2.moveWindow("Image", 300, 300)


cv2.waitKey(0)
cv2.destroyAllWindows()