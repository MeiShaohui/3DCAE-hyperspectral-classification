from __future__ import division, print_function, absolute_import

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv3D, Conv3DTranspose, PReLU, BatchNormalization, MaxPool3D
from keras import backend as K
from keras import regularizers
import keras
import numpy as np
import h5py
from sklearn import preprocessing
import math
from keras.utils import plot_model


# 224 -> 201 -> 178 -> 9 -> 201 -> 224
# def DCAE_v2(weight_decay=0.0005):
#     model = Sequential()
#     model.add(Conv3D(filters=24,
#                      input_shape=(224, 5, 5, 1),
#                      kernel_size=(24, 3, 3),
#                      strides=(1, 1, 1),
#                      kernel_regularizer=regularizers.l2(l=weight_decay),
#                      padding='valid', name="Conv1"))
#     model.add(BatchNormalization(name="BN1"))
#     model.add(PReLU(name="PReLU1"))

#     model.add(Conv3D(filters=48,
#                      kernel_size=(24, 3, 3),
#                      strides=(1, 1, 1),
#                      kernel_regularizer=regularizers.l2(l=weight_decay),
#                      padding='valid', name="Conv2"))
#     model.add(BatchNormalization(name="BN2"))
#     model.add(PReLU(name="PReLU2"))

#     model.add(MaxPool3D(pool_size=(18, 1, 1),
#                         strides=(18, 1, 1), name="Pool1"))

#     model.add(Conv3DTranspose(filters=24,
#                               kernel_size=(9, 3, 3),
#                               kernel_regularizer=regularizers.l2(
#                                   l=weight_decay),
#                               strides=(22, 1, 1), name="Deconv1", padding='valid'))
#     model.add(BatchNormalization(name="BN3"))
#     model.add(PReLU(name="PReLU3"))
#     model.add(Conv3DTranspose(filters=1,
#                               kernel_size=(27, 3, 3),
#                               kernel_regularizer=regularizers.l2(
#                                   l=weight_decay),
#                               strides=(1, 1, 1), name="Deconv2", padding='valid'))
#     model.add(BatchNormalization(name="BN4"))
#     return model


def DCAE_v2(weight_decay=0.0005):
    model = Sequential()
    model.add(Conv3D(filters=24,
                     input_shape=(200, 5, 5, 1),
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv1"))
    model.add(BatchNormalization(name="BN1"))
    model.add(PReLU(name="PReLU1"))

    model.add(Conv3D(filters=48,
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv2"))
    model.add(BatchNormalization(name="BN2"))
    model.add(PReLU(name="PReLU2"))

    model.add(MaxPool3D(pool_size=(18, 1, 1),
                        strides=(18, 1, 1), name="Pool1"))

    model.add(Conv3DTranspose(filters=24,
                              kernel_size=(9, 3, 3),
                              kernel_regularizer=regularizers.l2(
                                  l=weight_decay),
                              strides=(22, 1, 1), name="Deconv1"))
    model.add(BatchNormalization(name="BN3"))
    model.add(PReLU(name="PReLU3"))
    model.add(Conv3DTranspose(filters=1,
                              kernel_size=(25, 3, 3),
                              kernel_regularizer=regularizers.l2(
                                  l=weight_decay),
                              strides=(1, 1, 1), name="Deconv2"))
    model.add(BatchNormalization(name="BN4"))
    return model


def DCAE_v2_feature(weight_decay=0.0005):
    model = Sequential()
    model.add(Conv3D(filters=24,
                     input_shape=(200, 145, 145, 1),
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv1"))
    model.add(BatchNormalization(name="BN1"))
    model.add(PReLU(name="PReLU1"))

    model.add(Conv3D(filters=48,
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv2"))
    model.add(BatchNormalization(name="BN2"))
    model.add(PReLU(name="PReLU2"))

    model.add(MaxPool3D(pool_size=(18, 1, 1),
                        strides=(18, 1, 1), name="Pool1"))
    return model


def DCAE_LR(n_class=17, weight_decay=0.0005):
    model = Sequential()
    model.add(Conv3D(filters=24,
                     input_shape=(200, 5, 5, 1),
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv1", trainable=False))
    model.add(BatchNormalization(name="BN1", trainable=False))
    model.add(PReLU(name="PReLU1", trainable=False))

    model.add(Conv3D(filters=48,
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv2", trainable=True))
    model.add(BatchNormalization(name="BN2", trainable=True))
    model.add(PReLU(name="PReLU2", trainable=True))

    model.add(MaxPool3D(pool_size=(18, 1, 1), strides=(
        18, 1, 1), name="Pool1", trainable=True))
    model.add(Flatten(name="Flat1", ))
    #model.add(Dense(100, name="FC1", ))
    #model.add(PReLU(name="PReLU_FC", ))
    model.add(Dense(n_class, activation='softmax', name="FC2",
                    kernel_regularizer=regularizers.l2(l=weight_decay),))
    return model


def DCAE_LR_feature(n_class=17, weight_decay=0.0005):
    model = Sequential()
    model.add(Conv3D(filters=24,
                     input_shape=(200, 5, 5, 1),
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv1", trainable=False))
    model.add(BatchNormalization(name="BN1", trainable=False))
    model.add(PReLU(name="PReLU1", trainable=False))

    model.add(Conv3D(filters=48,
                     kernel_size=(24, 3, 3),
                     strides=(1, 1, 1),
                     kernel_regularizer=regularizers.l2(l=weight_decay),
                     padding='valid', name="Conv2", trainable=True))
    model.add(BatchNormalization(name="BN2", trainable=True))
    model.add(PReLU(name="PReLU2", trainable=True))

    model.add(MaxPool3D(pool_size=(18, 1, 1), strides=(
        18, 1, 1), name="Pool1", trainable=True))
    return model


if __name__ == '__main__':
    net = DCAE_v2()
    model = DCAE_v2(weight_decay=0.0005)
    model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['MSE'])
    plot_model(model, show_shapes=True, to_file='model_LR.png')
    model_2 = DCAE_v2_feature()
    plot_model(model_2, show_shapes=True, to_file='model_conv.png')
    model_3 = DCAE_v2_feature()
    plot_model(model_2, show_shapes=True, to_file='model_conv')
    json_string = net.to_json()
    with open('mlp_model.json', 'w') as of:
        of.write(json_string)
