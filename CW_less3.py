import cv2
import numpy as np


img = np.zeros((512,512,3), np.uint8)
#rgb = brg
#img[:] = 245, 66, 135

#img[100:150, 100:300] = 245, 66, 135 #перший аргумент по y, а другий по x

cv2.rectangle(img,(100,100),(200,200),(245, 66, 135),2)  #1 кортедж - ліва верхня координата, 2 кортедж права нижня
#3 кортедж показує колір, останній - товщина обводки (-2, cv2.fill() - заливка)

cv2.line(img,(100,100),(200,200),(245, 66, 135),2)
#Початкова точка (1 кортедж), 2 кортедж - кінцева точка, 3 кортедж - кортедж кольору, 4 - товщина лінії

print(img.shape)
# cv2.line(img, (0,0), (512, 512), (245, 66, 135),2)
# cv2.line(img, (0, 512), (512, 0), (245, 66, 135),2)

cv2.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (245, 66, 135),2)
cv2.line(img, (img.shape[1]//2, 0), (img.shape[0]//2, img.shape[0]), (245, 66, 135),2)

cv2.circle(img, (200, 200), 20, (245, 66, 135), 2)
cv2.putText(img, "Lyuta Mariia", (300, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (245, 66, 135), 1) #standart size of font
#300 - x, 200 - y
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
