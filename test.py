from keras import layers, models
from keras.utils import to_categorical
import numpy as np

input = layers.Input(shape=(10,))
layer = layers.Dense(1, activation='relu')(input)
output = layers.Dense(2, activation='relu')(layer)

model = models.Model(inputs=input, outputs=output)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


x = np.zeros((10, 10))
x[0:5, :] = 1
y = np.zeros(10)
y[0:5] = 1

y = to_categorical(y)
print(model.train_on_batch(x, y, return_model_output=True))
print(model.test_on_batch(x, y, return_model_output=True))

model.save('D:/test.h5')