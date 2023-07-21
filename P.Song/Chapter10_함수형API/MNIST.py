import tensorflow as tf
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

#1:
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

#2
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

#expand data with channel=1
x_train = np.expand_dims(x_train, axis=3)   #(60000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=3)     #(10000, 28, 28, 1)

#one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#3 build cnn
def create_cnn2d(input_shape, num_class=10):
    inputs=layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(units=10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_cnn2d(input_shape=x_train.shape[1:])

#train the model
#train model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
ret = model.fit(x_train, y_train, epochs=100, batch_size=400, verbose=0)

#evaluate the model
y_pred = model.predict(x_train)
y_label = np.argmax(y_pred, axis=1)
C = tf.math.confusion_matrix(np.argmax(y_train, axis=1), y_label)
print('confucion_matrix: ', C)

train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)