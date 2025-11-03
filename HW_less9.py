import cv2

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", 'data/MobileNet/mobileNet.caffemodel')

classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

files_to_process = ["cat.jpg",
                    "dog.jfif",
                    "phone.jpg",
                    "slon.jpg",
                    "stil.jpg"]

class_counts={}

for filename in files_to_process:
    image_path = f"images/MobileNet/{filename}"
    image = cv2.imread(image_path)

    if image is None:
        print(f"\nПОМИЛКА: Не вдалося прочитати файл {image_path}.")
        continue




    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    preds = net.forward()

    idx = preds[0].argmax()

    label = classes[idx] if idx < len(classes) else "Unknown"
    conf = float(preds[0][idx]) * 100

    print(f"\n--- Файл: {filename} ---")
    print(f"Клас: {label}")
    print(f"Впевненість: {conf:.2f}%")



    if label in class_counts:
        class_counts[label] += 1
    else:
        class_counts[label] = 1


    text = f'{label}: {int(conf)}%'
    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(f"Image: {filename}", image)
    cv2.waitKey(0)



cv2.destroyAllWindows()

print(f"\n ПІДСУМКОВА ТАБЛИЦЯ")
for label, count in class_counts.items():
    print(f"{label:<26} | {count}")