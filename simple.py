from keras import layers, models
import numpy as np

x = np.array([0, 1, 2, 3, 4])
y = x * 2 + 1

model = models.Sequential()
model.add(layers.Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x[:2], y[:2], epochs=1000, verbose=0)

print('Targets: ', y[2:])
print('Predictions: ', model.predict(x[2:]).flatten())
