

import tensorflow as tf
from keras import layers
from tensorflow.keras import models
import numpy as np
from keras.preprocessing import image


train_ds = tf.keras.preprocessing.image_dataset_from_directory('data/train',
                                                               image_size = (128,188),
                                                               batch_size = 30,
                                                               label_mode = "catrgorical")

test_ds = tf.keras.preprocessing.image_dataset_from_directory('data/test',
                                                               image_size = (128,188),
                                                               batch_size = 30,
                                                               label_mode = "catrgorical")

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

model = models.Sequental()

model.app(layers.Conv2D(
    filters=32,                 #кількість фільтрів
    kernel_size=(3,3),          #розмір фільтра
    activation='relu',          #функція активації
    input_shape=(128, 128, 3)   #форма вхідного зображення(RGB)
))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax')) #класи - 3

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

history = model.fit(train_ds, epochs = 50, validation_date = test_ds)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Правдивість: {test_acc}')

class_name = ['cars', 'cats', 'dogs']

img = image.load_img('images/1dog.jpg', target_size=(128,128))

image_array=image.img_to_array(img)
image_array = image_array/255.0
image_array=np.expand_dims(image_array, axis=0)

prediction=model.predict(image_array)
presict_index = np.argmax(prediction[0])


print(f'Імовірність по класам: {prediction[0]}')
print(f'Модель визначила: {class_name[presict_index]}')


