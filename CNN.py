from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

batch_size = 1280
nb_classes = 10
nb_epoch = 12

img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = (2,2)
kernel_size = (3,3)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
# 建立序贯模型
model = Sequential()
# 卷积层，初步提取特征
model.add(Convolution2D(nb_filters, kernel_size[0] ,kernel_size[1],border_mode='valid',input_shape=input_shape))
model.add(Activation('relu'))

# 卷积层，再次提取特征
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

#池化层，提取主要特征
model.add(MaxPooling2D(pool_size=pool_size))
#防止过拟合
model.add(Dropout(0.25))

#平坦层
model.add(Flatten())

#全连接层
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#全连接层
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#输出模型摘要
model.summary()

#编译模型
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

#训练模型
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_data=(X_test, Y_test))

#徐连以后的模型用于预测
score = model.evaluate(X_test, Y_test, verbose=0)


#输出预测结果
print('Test score:', score[0])
print('Test accuracy:', score[1])
