import cv2
import numpy as np

image = cv2.imread("images/me.jpg")
image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 200, 300)

kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)
image = cv2.erode(image, kernel, iterations=1)


cv2.imshow("me", image)
cv2.moveWindow("me", 100, 100)



gmail = cv2.imread('images/gmail.jpg')
gmail = cv2.resize(gmail, (gmail.shape[1]//6, gmail.shape[0]//6))
gmail = cv2.rotate(gmail, cv2.ROTATE_90_COUNTERCLOCKWISE)

gmail = cv2.cvtColor(gmail, cv2.COLOR_BGR2GRAY)
gmail = cv2.Canny(gmail, 350, 400)

# kernel = np.ones((5, 5), np.uint8)
# gmail = cv2.dilate(gmail, kernel, iterations=1)
# gmail = cv2.erode(gmail, kernel, iterations=1)

cv2.imshow("gmail", gmail)

cv2.waitKey(0)
cv2.destroyAllWindows()