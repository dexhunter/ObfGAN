from __future__ import print_function
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import os

os.makedirs("models", exist_ok=True)

batch_size = 128
num_classes = 10
epochs = 14

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def get_inference():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def get_obfmodel(neuron=100):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(neuron))
    model.add(Dense(28*28))
    model.add(Reshape(input_shape))
    return model

inference_model = get_inference()
inference_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint


inference_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[ModelCheckpoint('models/inf-best.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')],
          verbose=1,
          validation_data=(x_test, y_test))

neurons = [8, 16, 32, 64, 128, 256, 512]
for num in neurons:
    obfmodel = get_obfmodel(neuron=num)
    obfmodel.build((None, 28, 28, 1))
    infmodel = load_model('models/inf-best.h5')
    infmodel.trainable = False
    for l in infmodel.layers:
        l.trainable = False
    combined_model = Model(inputs=obfmodel.input, outputs=infmodel(obfmodel.output))
    combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(combined_model.summary())

    com_path = 'models/combined-best' + str(num) + '.h5'

    combined_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[ModelCheckpoint(com_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')],
              verbose=1,
              validation_data=(x_test, y_test))

for num in neurons:

    com_path = 'models/combined-best' + str(num) + '.h5'
    combined_model = load_model(com_path)
    score = combined_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])