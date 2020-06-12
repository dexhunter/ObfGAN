import os
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Add, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU, Reshape, Activation, MaxPooling2D
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import multi_gpu_model
import pickle as pkl
import datetime
# from keras_radam import RAdam
import tensorflow as tf
# set random seed
np.random.seed(1)
tf.random.set_seed(1)

gray = False
use_gpu = False
batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29
max_epochs = 1000

train_len = 87000
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)


with open("train2.pkl", "rb") as f:
    X_train, y_train = pkl.load(f)

def grayscale(data, dtype='float32'):
    r,g,b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r*data[:,:,:,0]+g*data[:,:,:,1]+b*data[:,:,:,2]
    rst = np.expand_dims(rst, axis=3)
    return rst

if gray:
    X_train = grayscale(X_train)
    target_dims = (imageSize, imageSize, 1)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)


# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_trainHot = to_categorical(y_train, num_classes=num_classes)
y_testHot = to_categorical(y_test, num_classes=num_classes)

# train_image_generator = ImageDataGenerator(
#     # samplewise_center=True,
#     # samplewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )
#
# val_image_generator = ImageDataGenerator(
#     # samplewise_center=True,
#     # samplewise_std_normalization=True,
# )
#
# train_generator = train_image_generator.flow(x=X_train, y=y_trainHot, batch_size=batch_size, shuffle=True)
# val_generator = val_image_generator.flow(x=X_test, y=y_testHot, batch_size=batch_size, shuffle=False)

def get_inference_model():
    model = Sequential()
    model.add(Flatten(input_shape=(target_dims)))

    for i in range(6):
        model.add(Dense(1024))
        model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def get_inference_model_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(target_dims)))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_obfgan():
    model = Sequential()
    model.add(Flatten(input_shape=(target_dims)))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(np.prod(target_dims)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Reshape(target_dims))
    return model

def get_obfgan_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(target_dims)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(np.prod(target_dims)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Reshape(target_dims))
    return model

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(target_dims)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, target_dims)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[target_dims]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes))

    return model


best_infmodel_path = "models/infmodel-best5.h5"
# infmodel_path = "models/infmodel-cnn.h5"
best_obf_path = "models/obfnet-best-fc128.h5"

if gray:
    best_infmodel_path = "models/infmodel-gray-best-cnn.h5"
    infmodel_path = "models/infmodel-gray-cnn.h5"

infmodelcheck_cb = ModelCheckpoint(best_infmodel_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# inference_model = get_inference_model_cnn()
inference_model = get_inference_model()
if use_gpu:
    inference_model = multi_gpu_model(inference_model, gpus=4)
inference_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
# inference_model.fit(x=X_train, y=y_trainHot, batch_size=batch_size, epochs=max_epochs, callbacks=[infmodelcheck_cb, tensorboard_cb], validation_data=(X_test, y_testHot))
inference_model.fit(x=X_train, y=y_trainHot, batch_size=batch_size, epochs=100, callbacks=[infmodelcheck_cb], validation_data=(X_test, y_testHot))
# inference_model.save(infmodel_path)

obfgan = get_obfgan()
obfgan.build((None,)+target_dims)
infmodel = load_model(best_infmodel_path)
infmodel.trainable = False
for l in infmodel.layers:
    l.trainable = False
combined_model = Model(inputs=obfgan.input, outputs=infmodel(obfgan.output))
if use_gpu:
    combined_model = multi_gpu_model(combined_model, gpus=4)
combined_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=["accuracy"])
callbacks = [ModelCheckpoint(best_obf_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'), tensorboard_cb]
callbacks = [ModelCheckpoint(best_obf_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]
combined_model.fit(x=X_train, y=y_trainHot, batch_size=batch_size, epochs=max_epochs, callbacks=callbacks, validation_data=(X_test, y_testHot))
print(combined_model.summary())
print(infmodel.evaluate(X_test, y_testHot, verbose=0))

# combined_model = load_model(best_obf_path)
# print(combined_model.summary())
# print(combined_model.evaluate(X_test, y_testHot, verbose=0))
#
# from tensorflow.keras import backend as K
# get_10th_layer_output = K.function([combined_model.layers[0].input], [combined_model.layers[10].output])
# layer_output = get_10th_layer_output([X_test])[0]
#
# from PIL import Image
# import cv2
#
# def get_concat_h(im1, im2):
#     dst =  Image.new('RGB', (im1.width + im2.width, im1.height))
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (im1.width, 0))
#     return dst
#
# h1 = []
# h2 = []
#
# for i in range(0, 11):
#     # h1 = Image.fromarray(X_test[i], 'RGB')
#     # h2 = Image.fromarray(layer_output[i], 'RGB')
#     # get_concat_h(h1, h2).save('imgs/c'+str(i)+'.jpg')
#     h1.append(X_test[i])
#     h2.append(layer_output[i])
#
# v1 = np.concatenate(np.array(h1), axis=1)
# v2 = np.concatenate(np.array(h2), axis=1)
# v = np.concatenate((v1, v2), axis=0)
# cv2.imwrite('imgs/cat.jpg', v)
# cv2.imwrite('imgs/cat1.jpg', v1)
# cv2.imwrite('imgs/cat2.jpg', v2)
