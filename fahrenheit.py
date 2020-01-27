from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np

celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.zeros([7, 1], dtype=float)

for i, c in enumerate(celsius_q):
    fahrenheit_a[i] = c * 1.8 + 32

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))
#create a layer (neurons, inputs)
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
#put together layers, "create neural net"
model = tf.keras.Sequential([l0])
#loss function, optimizer, learning rate
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
#fit = train
history = model.fit(celsius_q, fahrenheit_a, epochs=2000, verbose=False)
print("Finished training the model")
#plot loss
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()
#predict
# print(model.predict([100.0]))
celsius_test = np.array([-50,-5,1,25,41],dtype=float)

for i, c in enumerate(celsius_test):
    print("Cel {} , Fa {} , Pred {}".format(c, c*1.8 +32,model.predict([c])))

#get weights
print("These are the layer variables: {}".format(l0.get_weights()))
