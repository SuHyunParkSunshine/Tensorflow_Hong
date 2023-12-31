from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt

# 모델 훈련
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# 모델 컴파일
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# 데이터 전처리와 증강 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, # 회전을 몇 도 시킬것인가
    width_shift_range=0.1, # 수평으로 평행 이동
    height_shift_range=0.1, # 수직으로 평행 이동
    shear_range=0.4, # y축 방향으로 각도 증가
    zoom_range=0.5, # 확대/축소 범위
    horizontal_flip=True, # 좌우 대칭
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "C:/K-Digital3/TenserflowHong/P.Song/CNN/CnnData/train",
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

valid_generator = test_datagen.flow_from_directory(
    "C:/K-Digital3/TenserflowHong/P.Song/CNN/CnnData/test",
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# 모델 훈련
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=valid_generator,
    validation_steps=50,
)

model.save('C:/K-Digital3/TenserflowHong/P.Song/CNN/models_saved/cat_and_dog.h5')

# 시각화
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.suptitle('Accuracy & Loss')
plt.tight_layout()

plt.show()