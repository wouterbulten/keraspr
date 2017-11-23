from keras import layers, models
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np

input = layers.Input(shape=(10,))
layer = layers.Dense(1, activation='relu')(input)
output = layers.Dense(2, activation='relu')(layer)

model = models.Model(inputs=input, outputs=output)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'],
              return_model_output=True)


x = np.zeros((100, 10))
x[0:50, :] = 1
y = np.zeros(100)
y[0:50] = 1

y = to_categorical(y)

# Try normal functions
print(model.train_on_batch(x, y))
print(model.test_on_batch(x, y))

# Now try fitting
model.fit(x, y, batch_size=2, verbose=True)


model.save('D:/test.h5')
del model

# Load model
model2 = load_model('D:/test.h5')
print(model2.train_on_batch(x, y))
print(model2.test_on_batch(x, y))