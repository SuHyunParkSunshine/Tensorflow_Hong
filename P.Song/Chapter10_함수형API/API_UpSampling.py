import tensorflow as tf
import numpy as np

#1:

#2: create 2D input data
A = np.array([[1, 2, 3, 4], 
              [5, 6, 7, 8]], dtype='float32')
A = A.reshape(1, 2, 4, 1)   # (batch, rows, cols, channels)

#3: build a model
x = tf.keras.layers.Input(shape=A.shape[1:])
y = tf.keras.layers.UpSampling2D()(x)   #size=(2,2), [None, 4, 8, 1]

u = tf.keras.layers.Reshape([8, 1])(x)
z = tf.keras.layers.UpSampling1D()(u) #size=2, [None, 16, 1]

model = tf.keras.Model(inputs=x, outputs=[y, z])
model.summary()

#4: apply A to model
##output = model(A)     #Tensoroutput
output = model.predict(A)
print("A[0,:,:,0]=", A[0,:,:,0])
print("output[0]=", output[0][0,:,:,0]) #y
print("output[1]=", output[1][0,:,0]) #z