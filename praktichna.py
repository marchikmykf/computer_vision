import cv2
import numpy as np

img = np.zeros((400,600,3), np.uint8)
img[:] = (240, 253, 255)

photo = cv2.imread("images/foto.jpg")
photo = cv2.resize(photo, (120, 120))
img[30:150, 30:150] = photo       #1- y, 2-x

qr = cv2.imread("images/frame.png")
qr = cv2.resize(qr, (qr.shape[1]//3, qr.shape[0]//3))
img[230:330, 450:550] = qr

cv2.rectangle(img,(10,10),(590,390),(133, 141, 143),4)

cv2.putText(img, "Lyuta Mariia", (170, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (31, 33, 33), 2)
cv2.putText(img, "Computer Vision Student", (170, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, (133, 141, 143), 2)
cv2.putText(img, "Email: maria.liuta0206@gmail.com", (170, 210), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (2,142,163), 1)
cv2.putText(img, "Phone: +380974162511", (170, 250), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (2,142,163), 1)
cv2.putText(img, "02/06/2010", (170, 290), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (2,142,163), 1)
cv2.putText(img, "OpenCV Business Card", (170, 360), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (31, 33, 33), 2)


cv2.imshow("Image", img)
cv2.imwrite("business_card.png", img)
cv2.waitKey(0)
cv2.destroyAllWindows()