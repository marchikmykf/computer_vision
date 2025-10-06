import cv2
import numpy as np

img=cv2.imread("images/saay.jpg")
scale = 1
img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
print(img.shape)


img_copy=img.copy()
img_copy_colour = img_copy
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy=cv2.GaussianBlur(img_copy, (5,5), 2)

#підсилення контрасту
img_copy=cv2.equalizeHist(img_copy)

img_copy=cv2.Canny(img_copy, 100, 100)

contours, hierarchy=cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.RETR_EXTERNAL - шукає зовнішні краї
#cv2.CHAIN_APPROX_SIMPLE - апроксимація (вираження одих об'єктів через інші), шукає крайні точки

#малювання контурів через прямокутник
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100: #фільтер шуму
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img_copy_colour, [cnt], -1, (0, 255, 0), 2)
        cv2.rectangle(img_copy_colour, (x,y), (x+w, y+h), (0,255,0), 2)
        text_y= y - 5 if y - 5 > 10 else y +15
        text = f'x:{x}, y:{y}, S:{int(area)}'
        cv2.putText(img_copy_colour, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)



# cv2.imshow("img", img)
# cv2.moveWindow("img", 100, 100)
# cv2.imshow("img_copy", img_copy)
# cv2.moveWindow("img_copy", 700, 100)
cv2.imshow("copy", img_copy_colour)
cv2.waitKey(0)
cv2.destroyAllWindows()