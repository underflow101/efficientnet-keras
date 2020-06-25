###############################################################################
# model.py
#
# Dev. Dongwon Paek
# Description: EfficientNet B0 model source code
#              This source code only works with efficientnet.py source code
#              This source code is different from original paper due to its
#              procedure code format
# Changes: Original EfficientNet B0 Model is designed into subclassed
#          architecture. But this source code aims to build exactly same model
#          into functional model.
#          This is due to its unableness of applying tfmot and pruning method
#          on subclassed architecture model.
###############################################################################

###############################################################################
# Original Paper
#
# Mingxing Tan, and Quoc V.Le, EfficientNet: Rethinking Model Scaling for
# Convolutional Neural Networks, ICML 2019
###############################################################################

import h5py
import os, shutil
import matplotlib.pyplot as plt
import math
from copy import deepcopy

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, Concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax

#from keras.models import Model
#from keras.layers import Layer, Conv2D, BatchNormalization, DepthwiseConv2D
#from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, Concatenate
#from keras import Sequential
#from keras.activations import softmax

from utils import round_filters, round_repeats
from hyperparameter import NUM_CLASSES, EPOCHS, BATCH_SIZE, WIDTH, HEIGHT, CHANNELS

def SEBlock(inputs, input_channels, ratio=0.25):
    num_reduced_filters = max(1, int(input_channels * ratio))
    branch = GlobalAveragePooling2D()(inputs)
    branch = tf.expand_dims(input=branch, axis=1)
    branch = tf.expand_dims(input=branch, axis=1)
    branch = Conv2D(filters=num_reduced_filters, kernel_size=(1,1), strides=1, padding='same')(branch)
    branch = tf.nn.swish(branch)
    branch = Conv2D(filters=input_channels, kernel_size=(1,1), strides=1, padding='same')(branch)
    branch = tf.nn.sigmoid(branch)
    
    output = inputs * branch
    print("SEBlock output's type: ", end='')
    print(type(output))
    return output

def MBConvLayer(in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
    conv1 = Conv2D(filters=in_channels * expansion_factor, kernel_size=(1,1), strides=1, padding='same', use_bias=False)
    bn1 = BatchNormalization()
    dwconv = DepthwiseConv2D(kernel_size=(k,k), strides=stride, padding='same', use_bias=False)
    bn2 = BatchNormalization()
    se = SEBlock(input_channels=in_channels * expansion_factor)
    conv2 = Conv2D(filters=out_channels, kernel_size=(1,1), strides=1, padding='same', use_bias=False)
    bn3 = BatchNormalization()
    dropout = Dropout(rate=drop_connect_rate)

def MBConv(inputs, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, training=None, **kwargs):
    ret = Sequential()
    ret.add(Conv2D(filters=in_channels * expansion_factor, kernel_size=(1,1), strides=1, padding='same', use_bias=False))
    ret.add(BatchNormalization())
    print("inputs = ", end='')
    print(inputs)
    print("in_channels = ", end='')
    print(in_channels)
    print("out_channels = ", end='')
    print(out_channels)
    print("MBConv - Conv2D 1")
    x = Conv2D(filters=in_channels * expansion_factor, kernel_size=(1,1), strides=1, padding='same', use_bias=False)(inputs)
    print("MBConv - BatchNormalization 1")
    x = BatchNormalization()(x, training=training)
    print("MBConv - swish 1")
    x = tf.nn.swish(x)
    print("MBConv - DepthwiseConv2D 1")
    x = DepthwiseConv2D(kernel_size=(k,k), strides=stride, padding='same', use_bias=False)(x)
    print("MBConv - BatchNormalization 2")
    x = BatchNormalization()(x, training=training)
    print("MBConv - SEBlock")
    x = SEBlock(inputs=x, input_channels=in_channels * expansion_factor)
    print("MBConv - swish 2")
    x = tf.nn.swish(x)
    print("MBConv - Conv2D 2")
    x = Conv2D(filters=out_channels, kernel_size=(1,1), strides=1, padding='same', use_bias=False)(x)
    print("MBConv - BatchNormalization 3")
    x = BatchNormalization()(x, training=training)
    
    print("MBConv - Dropout?")
    if stride == 1 and in_channels == out_channels:
        print("in first if statement!")
        if drop_connect_rate:
            print("MBConv - Dropout!")
            x = Dropout(rate=drop_connect_rate)(x, training=training)
        print("MBConv - Layer ADD")
        x = tf.keras.layers.add([x, inputs])
    
    ret.add(x)
    print(type(x))
    
    return x

def MBConvBlock(inputs, in_channels, out_channels, layer, stride, expansion_factor, k, drop_connect_rate):
    block = Sequential()
    print("layer = ", end='')
    print(layer)
    
    for i in range(layer):
        print("i = ", end='')
        print(i)
        if i == 0:
            print("Final layer add! Start!")
            block.add(MBConv(inputs=inputs, in_channels=in_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                             stride=stride, k=k, drop_connect_rate=drop_connect_rate))
            print("Final layer add! Done!!")
        else:
            print("%dth Layer add! Start!" % i)
            block.add(MBConv(inputs=inputs, in_channels=out_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                             stride=1, k=k, drop_connect_rate=drop_connect_rate))
            print("%dth Layer add! DONE!" % i)
    return block

def EfficientNet(input_shape, classes=4, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=224, drop_connect_rate=0.2,
                 training=None, mask=None):
    #inputs = Input(shape=input_shape)
    inputs = tf.keras.Input(shape=input_shape)
    
    #x = Conv2D(filters=round_filters(32, width_coefficient), kernel_size=(3,3), strides=2, padding='same', use_bias=False)(inputs)
    #x = BatchNormalization()(x, training=training)
    #x = tf.nn.swish(x)
    
    # MBConvBlock1
    print("MBConvBlock1 !!!!!!!!!!!!!!!!!!!")
    block1 = MBConvBlock(inputs=inputs, in_channels=round_filters(32, width_coefficient), out_channels=round_filters(16, width_coefficient),
                    layer=round_repeats(1, depth_coefficient), stride=1, expansion_factor=1, k=3,
                    drop_connect_rate=drop_connect_rate)
    
    # MBConvBlock2
    print("MBConvBlock2 !!!!!!!!!!!!!!!!!!!")
    block2 = MBConvBlock(in_channels=round_filters(16, width_coefficient), out_channels=round_filters(24, width_coefficient),
                    layer=round_repeats(2, depth_coefficient), stride=2, expansion_factor=6, k=3,
                    drop_connect_rate=drop_connect_rate)
    
    # MBConvBlock3
    print("MBConvBlock3 !!!!!!!!!!!!!!!!!!!")
    block3 = MBConvBlock(in_channels=round_filters(24, width_coefficient), out_channels=round_filters(40, width_coefficient),
                    layer=round_repeats(2, depth_coefficient), stride=2, expansion_factor=6, k=5,
                    drop_connect_rate=drop_connect_rate)
    
    # MBConvBlock4
    print("MBConvBlock4 !!!!!!!!!!!!!!!!!!!")
    block4 = MBConvBlock(in_channels=round_filters(40, width_coefficient), out_channels=round_filters(80, width_coefficient),
                    layer=round_repeats(3, depth_coefficient), stride=2, expansion_factor=6, k=3,
                    drop_connect_rate=drop_connect_rate)
    
    # MBConvBlock5
    print("MBConvBlock5 !!!!!!!!!!!!!!!!!!!")
    block5 = MBConvBlock(in_channels=round_filters(80, width_coefficient), out_channels=round_filters(112, width_coefficient),
                    layer=round_repeats(3, depth_coefficient), stride=1, expansion_factor=6, k=5,
                    drop_connect_rate=drop_connect_rate)
    
    # MBConvBlock6
    print("MBConvBlock6 !!!!!!!!!!!!!!!!!!!")
    block6 = MBConvBlock(in_channels=round_filters(112, width_coefficient), out_channels=round_filters(192, width_coefficient),
                    layer=round_repeats(4, depth_coefficient), stride=2, expansion_factor=6, k=5,
                    drop_connect_rate=drop_connect_rate)
    
    # MBConvBlock7
    print("MBConvBlock7 !!!!!!!!!!!!!!!!!!!")
    block7 = MBConvBlock(in_channels=round_filters(192, width_coefficient), out_channels=round_filters(320, width_coefficient),
                    layer=round_repeats(1, depth_coefficient), stride=1, expansion_factor=6, k=3,
                    drop_connect_rate=drop_connect_rate)
    
    x = Conv2D(filters=round_filters(32, width_coefficient), kernel_size=(3,3), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x, training=training)
    x = tf.nn.swish(x)
    
    x = block1(x)
    x = block2(x)
    x = block3(x)
    x = block4(x)
    x = block5(x)
    x = block6(x)
    x = block7(x)
    
    x = Conv2D(filters=round_filters(1280, width_coefficient), kernel_size=(1,1), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x, training=training)
    x = tf.nn.swish(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=dropout_rate)(x, training=training)
    x = Dense(units=NUM_CLASSES, activation=softmax)(x)
    
    return Model(inputs=inputs, outputs=x)