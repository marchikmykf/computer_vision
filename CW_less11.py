import pandas as pd #csv
import numpy as np #
import tensorflow as tf #create neyronky
from tensorflow import keras #частина тф, створює шари та працює -з ними
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder  # переводить назви в числа
import matplotlib.pyplot as plt #вивчити

#2-зчитуємо інформацію з таблиці csv
df = pd.read_csv('data/figures.csv')
#df-datafile
print(df.head())

#3 - перетворюємо фігури в цифри
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label']) #create a new column

X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']

#4
model = keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=(3,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(8, activation='softmax'),
])

#5- компаляція моделі

model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
#adam - вибирає кращий алгоритм для навчання

#6-навчання

history = model.fit(X, y, epochs = 300, verbose = 0)
plt.plot(history.history['loss'], label='Втрати')
plt.plot(history.history['accuracy'], label='Точність')
plt.xlabel('Епоха')
plt.ylabel('Значення')
plt.title("Процес навчання моделі")
plt.legend()
plt.show()


#7-тестування

test = np.array([[25, 20, 0]])
pred=model.predict(test)

print(f'Імовірність кожного класу {pred}')
print(f'Модель визначила {encoder.inverse_transform([np.argmax])}')