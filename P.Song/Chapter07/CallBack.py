import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() # mnist는 손글씨 데이터

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # input layer ('28 by 28' -> 픽셀수)
model.add(tf.keras.layers.Dense(units=5, activation='sigmoid')) # hidden layer(노드 수: 5개)
model.add(tf.keras.layers.Dense(units=10, activation='softmax')) # output layer(노드 수: 10개)

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01) # learning_rate: 학습률(증감 범위)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜벡 설정: 특정 조건에서 모델 조기 종료 설정
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0.001, # 최소 갱신 값: validation loss 갱신 값이 0.001 이하가 되면 종료
                                            patience=1, # 위반 허용 횟수
                                            verbose=1, # callback 메세지 출력
                                            mode='auto') # mode에 min이 주어지면 모니터링 값 감수할 때 stopping/ max가 주어지면 모니터링 값 증가할때 stopping
# min_delta : 최소 갱신값(w 조정치가 min_delta범위까지 갈때) / patience : 위반 허용 횟수 / verbose : callback 메시지 출력 / mode : auto, min, max (각 해당될때 mode에서 값이 주어지면 모니터링 값 증가, 감소 할때 stop)

# 모델 학습
ret = model.fit(x_train, y_train, epochs=100, batch_size=200, #batch_size 한번의 fork에서 200개를 처리
                validation_split=0.2, verbose=2, callbacks=[callback]) # callback을 안 주면 100번 돌음
# batch_size : 연산 한번에 들어가는 데이터 크기 / validation_split : 검증 데이터로 사용할 학습 데이터의 비율 / epochs: 동작 세트 횟수

# 모델 학습 스케쥴러 정의하기
def scheduler(epoch, lr):
    if epoch % 2 == 0 and epoch: # 2번의 fork가 일어날떄마다
        return 0.1*lr # 0.1씩 학습률을 감소시키는 것
    return lr
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

ret = model.fit(x_train, y_train, epochs=10, batch_size=200,
                validation_split=0.2, verbose=0, callbacks=[callback])

path = "C:/K-Digital3/TenserflowHong/P.Song/tensorboard/"
if not os.path.isdir(path):
    os.mkdir(path)
logdidr = path + "3101"

callback = tf.keras.callbacks.TensorBoard(log_dir=logdidr, update_freq='epoch',
                                          histogram_freq=10, write_images=True)

ret = model.fit(x_train, y_train, epochs=100, batch_size=200,
                validation_split=0.2, verbose=2, callbacks=[callback])