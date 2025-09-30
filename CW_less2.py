import cv2
import numpy as np

# image = cv2.imread('images/flower.jpg')
# #image = cv2.resize(image, (600, 300))
# image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
# print(image.shape)
#
# #image=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# #image = cv2.flip(image, 0)
#
# image = cv2.GaussianBlur(image, (5, 5), 0) #в кортеджі можна ставити лише непарні, sigmaX - рівень потужності
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 180, 200)
#
# kernel = np.ones((5, 5), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# image = cv2.erode(image, kernel, iterations=1)
#
# print(image.shape)
#
#
# cv2.imshow('flower', image)
#cv2.imshow('flower', image[0:150, 0:100])

#video  = cv2.VideoCapture('video/sea.mov')
video  = cv2.VideoCapture(0)

while True:
    success, frame = video.read()
    if not success:  # якщо немає кадру — виходимо з циклу
        print("Не вдалося зчитати кадр (кінець відео або неправильний шлях).")
        break
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# cv2.waitKey(0)
cv2.destroyAllWindows()