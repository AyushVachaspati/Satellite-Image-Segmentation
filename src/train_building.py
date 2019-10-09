from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Conv2DTranspose, Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras import backend as K
import keras
import h5py
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam
from keras.backend import binary_crossentropy
import matplotlib.pyplot as plt
import datetime
import os
from sklearn.model_selection import train_test_split
import random

from keras.models import model_from_json

img_rows = 288
img_cols = 288


smooth = 1e-12

num_channels = 3
num_mask_channels = 1


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0,1,2])
    sum_ = K.sum(y_true + y_pred, axis=[0,1,2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis= [0,1,2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0,1,2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
   return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
    
 


def get_unet_deep():
    
    inputs = Input((img_rows, img_cols,num_channels))
    conv1 = Conv2D(filters=64, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(inputs) #xavier initilizer alias
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Conv2D(filters=64, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Conv2D(filters=128, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Conv2D(filters=256, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv3)

    conv4 = Conv2D(filters=512, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Conv2D(filters=512, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv4)

    conv5 = Conv2D(filters=1024, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Conv2D(filters=1024, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2),data_format="channels_last")(conv5)
    
    conv51 = Conv2D(filters=2048, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(pool5)
    conv51 = BatchNormalization()(conv51)
    conv51 = keras.layers.advanced_activations.ELU()(conv51)
    conv51 = Conv2D(filters=2048, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv51)
    conv51 = BatchNormalization()(conv51)
    conv51 = keras.layers.advanced_activations.ELU()(conv51)
    
    up61 = Conv2DTranspose(filters = 1024,kernel_size = (2,2),padding='valid', strides = (2,2), data_format="channels_last")(conv51)
    up61 = Concatenate(axis=-1)([up61, conv5])
    conv61 = Conv2D(filters=1024, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(up61)
    conv61 = BatchNormalization()(conv61)
    conv61 = keras.layers.advanced_activations.ELU()(conv61)
    conv61 = Conv2D(filters=1024, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv61)
    conv61 = BatchNormalization()(conv61)
    conv61 = keras.layers.advanced_activations.ELU()(conv61)
    
    
    up6 = Conv2DTranspose(filters = 512,kernel_size = (2,2),padding='valid', strides = (2,2), data_format="channels_last")(conv61)
    up6 = Concatenate(axis=-1)([up6, conv4])
    conv6 = Conv2D(filters=512, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Conv2D(filters=512, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = Conv2DTranspose(filters = 256,kernel_size = (2,2),padding='valid', strides = (2,2), data_format="channels_last")(conv6)
    up7 = Concatenate(axis=-1)([up7, conv3])
    conv7 = Conv2D(filters=256, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Conv2D(filters=256, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = Conv2DTranspose(filters = 128,kernel_size = (2,2),padding='valid', strides = (2,2), data_format="channels_last")(conv7)
    up8 = Concatenate(axis=-1)([up8, conv2])
    conv8 = Conv2D(filters=128, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Conv2D(filters=128, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = Conv2DTranspose(filters = 64,kernel_size = (2,2),padding='valid', strides = (2,2), data_format="channels_last")(conv8)
    up9 = Concatenate(axis=-1)([up9, conv1])
    conv9 = Conv2D(filters=64, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Conv2D(filters=64, kernel_size=(3,3),padding='same', kernel_initializer='glorot_uniform',data_format="channels_last")(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv10 = Conv2D(filters=num_mask_channels, kernel_size=(1,1),padding='same', 
                    kernel_initializer='glorot_uniform',activation='sigmoid',data_format="channels_last")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    print(model.summary())
    return model


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def form_batch(images,masks,batch_size):
    """images is 4D np array of shape [num_images,height,width,num_channels]
       masks is 4D np array of shape [num_images,height,width,mask_channels].
    """
    x_batch = np.zeros((batch_size,img_rows,img_cols,num_channels))
    y_batch = np.zeros((batch_size,img_rows,img_cols,num_mask_channels))
    images_height = images.shape[1]
    images_width = images.shape[2]

    for i in range(batch_size):
        random_width = random.randint(0, images_width - img_cols - 1)
        random_height = random.randint(0, images_height - img_rows - 1)
        random_image = random.randint(0, images.shape[0] - 1)
        y_batch[i] = np.array(masks[random_image,
                                    random_height: random_height + img_rows,
                                    random_width: random_width + img_cols,:])
        x_batch[i] = np.array(images[random_image,
                                     random_height: random_height + img_rows,
                                     random_width: random_width + img_cols,:])
    return x_batch, y_batch




def batch_generator(X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 0)
                    yb = flip_axis(yb, 0)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(0, 1)
                    yb = yb.swapaxes(0, 1)

            X_batch[i] = xb
            y_batch[i] = yb

        yield X_batch, y_batch


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)



def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model


if __name__ == '__main__':
    data_path = 'FP_Mask_3'
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet_deep()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
    f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')

    X_train = f['train']
    X_train = np.array(X_train)
    y_train = np.array(f['train_mask'])
    y_train = np.array(y_train)
    train_ids = np.array(f['train_ids'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    batch_size = 8
    nb_epoch = 50
    iters_per_epoch = 500

    suffix = 'buildings_3_'
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
                  steps_per_epoch= iters_per_epoch,
                  epochs=nb_epoch,
                  verbose=1,
                  validation_data=batch_generator(X_test, y_test, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
                  validation_steps = 50)
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    print(history.history.keys())
    # summarize history for accuracy
    fig = plt.figure()
    plt.plot(history.history['jaccard_coef_int'])
    plt.plot(history.history['val_jaccard_coef_int'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig("accuracy.pdf", bbox_inches='tight')
    
    # summarize history for loss
    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    fig.savefig("loss.pdf", bbox_inches='tight')
    f.close()

