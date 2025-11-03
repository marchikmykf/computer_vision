import cv2

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", 'data/MobileNet/mobileNet.caffemodel')
#upload model


#2- зчитуємо список назв класів

classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


#3 - вантажимо зображення
image = cv2.imread("images/MobileNet/dog.jfif")
image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))


#4- адаптуємо зображення для нейронки
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (224, 224)),
    1.0 / 127.5,
    (224, 224),
    (127.5, 127.5, 127.5)
)


#5 - кладемо зображення в мережу та запускаємо
net.setInput(blob)
preds = net.forward() #вектор ймовірності для наших класів

#6 крок - знаходимо індекс класу з найбільшою імовірністю
idx = preds[0].argmax()

#7 крок - дістаємо назву класу та впевненість (точність) у відсотках
label = classes[idx] if idx < len(classes) else "Unknown"
conf = float(preds[0][idx]) * 100

#8 - виводимо результат в консоль
print("Class name: ", label)
print("Confidence: ", conf)

#9 - підписуємо зображення
text = f'{label}: {int(conf)}%'
cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()