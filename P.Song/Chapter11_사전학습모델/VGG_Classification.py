import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras import applications
from keras.preprocessing import image

gpus = tf.config.experimental.list_physical_devices('GPU')

#2
model = VGG16(weights='imagenet', include_top=True)
model.summary()

#3: predict an image
img_path = "C:/K-Digital3/TenserflowHong/P.Song/Chapter11_사전학습모델/DATA/elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
output = model.predict(x)

print('Predicted: ', decode_predictions(output, top=5)[0])

plt.imshow(img)
plt.axis("off")
plt.show()