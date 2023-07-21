import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

#2
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]], dtype=np.float32)
y_and = np.array([[0], [0], [0], [1]], dtype=np.float32)    #AND
y_or = np.array([[0], [1], [1], [1]], dtype=np.float32)     #OR
y_xor = np.array([[0], [1], [1], [0]], dtype=np.float32)

y_and = tf.keras.utils.to_categorical(y_and)
y_or = tf.keras.utils.to_categorical(y_or)
y_xor = tf.keras.utils.to_categorical(y_xor)

#3: build a model
x_and = layers.Input(shape=(2,))
x = layers.Dense(units=2, activation='sigmoid')(x_and)
out_and = layers.Dense(units=2, activation='softmax', name='and')(x)

x_or = layers.Input(shape=(2,))
x = layers.Dense(units=2, activation='sigmoid')(x_or)
out_or = layers.Dense(units=2, activation='softmax', name='or')(x)

x_xor = layers.Input(shape=(2,))
x = layers.Dense(units=2, activation='sigmoid')(x_xor)
out_xor = layers.Dense(units=2, activation='softmax', name='xor')(x)

model = tf.keras.Model(inputs=[x_and, x_or, x_xor],
                       outputs=[out_and, out_or, out_xor])
model.summary()

#4: train and evaluate
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1)
model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
ret = model.fit(x=[X, X, X], y=[y_and, y_or, y_xor],
                epochs=100, batch_size=4, verbose=0)

test = model.evaluate(x=[X, X, X], y=[y_and, y_or, y_xor], verbose=0)
print('total loss=', test[0])
print('AND: loss={}, acc={}'.format(test[1], test[4]))
print('OR: loss={}, acc={}'.format(test[2], test[5]))
print('XOR: loss={}, acc={}'.format(test[3], test[6]))

#5: draw graph
plt.plot(ret.history['loss'], 'k--', label='loss')
plt.plot(ret.history['and_loss'], 'r--', label='and_loss')
plt.plot(ret.history['or_loss'], 'g--', label='or_loss')
plt.plot(ret.history['or_loss'], 'b--', label='or_loss')

plt.plot(ret.history['and_accuracy'], 'r-', label='and_accuracy')
plt.plot(ret.history['or_accuracy'], 'g-', label='or_accuracy')
plt.plot(ret.history['xor_accuracy'], 'b-', label='xor_accuracy')

plt.xlabel('epochs')
plt.ylabel('loss and accuracy')
plt.legend(loc = 'best')
plt.show()