###############################################################################
# model.py
#
# Dev. Dongwon Paek
# Description: EfficientNet B0 model source code
#              This source code only works with efficientnet.py source code
###############################################################################

###############################################################################
# Original Paper
#
# Mingxing Tan, and Quoc V.Le, EfficientNet: Rethinking Model Scaling for
# Convolutional Neural Networks, ICML 2019
###############################################################################

import h5py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax

from utils import round_filters, round_repeats
from hyperparameter import NUM_CLASSES, EPOCHS, BATCH_SIZE, WIDTH, HEIGHT, CHANNELS

'''
@class: SEBlock
    - SEBlock stands for "Squeeze & Excitation" Block
    - This block reduces and expands the layer for model size
    - For reusing SEBlock, it is designed in class form
'''
class SEBlock(Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = GlobalAveragePooling2D()
        self.reduce_conv = Conv2D(filters=self.num_reduced_filters, kernel_size=(1,1), strides=1, padding='same')
        self.expand_conv = Conv2D(filters=input_channels, kernel_size=(1,1), strides=1, padding='same')
    
    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = tf.nn.swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output

'''
@class: MBConv
    - MBConv stands for Mobile-inverted Bottleneck Convolution Layer
    - MBConv is core layer in EfficientNet, as it appears 7 times in EfficientNet B0 from original paper
    - For reusing MBConv, it is designed in class form
'''
class MBConv(Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        print("ONLY INIT PROCESS")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        
        self.conv1 = Conv2D(filters=in_channels * expansion_factor, kernel_size=(1,1), strides=1, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.dwconv = DepthwiseConv2D(kernel_size=(k,k), strides=stride, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = Conv2D(filters=out_channels, kernel_size=(1,1), strides=1, padding='same', use_bias=False)
        self.bn3 = BatchNormalization()
        self.dropout = Dropout(rate=drop_connect_rate)
    
    def call(self, inputs, training=None, **kwargs):
        print("CALL START!!!!")
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = tf.nn.swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        print("THIS IS TYPE!!!", end=' ')
        print(type(x))
        return x
        
def MBConvBlock(in_channels, out_channels, layer, stride, expansion_factor, k, drop_connect_rate):
    block = Sequential()
    
    for i in range(layer):
        print("BLOCK ADD PROCEDURE")
        if i == 0:
            block.add(MBConv(in_channels=in_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                             stride=stride, k=k, drop_connect_rate=drop_connect_rate))
        else:
            block.add(MBConv(in_channels=out_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                             stride=1, k=k, drop_connect_rate=drop_connect_rate))
    return block

'''
@class: EfficientNet
    - Main source code for EfficientNet Model
    - Functional tf.keras model
'''
class EfficientNet(tf.keras.Model):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2):
        super(EfficientNet, self).__init__()
        print("EFFICIENTNET CLASS __INIT__")
        self.conv1 = Conv2D(filters=round_filters(32, width_coefficient), kernel_size=(3,3), strides=2, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        
        self.block1 = MBConvBlock(in_channels=round_filters(32, width_coefficient), out_channels=round_filters(16, width_coefficient),
                                  layer=round_repeats(1, depth_coefficient), stride=1, expansion_factor=1, k=3,
                                  drop_connect_rate=drop_connect_rate)
        self.block2 = MBConvBlock(in_channels=round_filters(16, width_coefficient), out_channels=round_filters(24, width_coefficient),
                                  layer=round_repeats(2, depth_coefficient), stride=2, expansion_factor=6, k=3,
                                  drop_connect_rate=drop_connect_rate)
        self.block3 = MBConvBlock(in_channels=round_filters(24, width_coefficient), out_channels=round_filters(40, width_coefficient),
                                  layer=round_repeats(2, depth_coefficient), stride=2, expansion_factor=6, k=5,
                                  drop_connect_rate=drop_connect_rate)
        self.block4 = MBConvBlock(in_channels=round_filters(40, width_coefficient), out_channels=round_filters(80, width_coefficient),
                                  layer=round_repeats(3, depth_coefficient), stride=2, expansion_factor=6, k=3,
                                  drop_connect_rate=drop_connect_rate)
        self.block5 = MBConvBlock(in_channels=round_filters(80, width_coefficient), out_channels=round_filters(112, width_coefficient),
                                  layer=round_repeats(3, depth_coefficient), stride=1, expansion_factor=6, k=5,
                                  drop_connect_rate=drop_connect_rate)
        self.block6 = MBConvBlock(in_channels=round_filters(112, width_coefficient), out_channels=round_filters(192, width_coefficient),
                                  layer=round_repeats(4, depth_coefficient), stride=2, expansion_factor=6, k=5,
                                  drop_connect_rate=drop_connect_rate)
        self.block7 = MBConvBlock(in_channels=round_filters(192, width_coefficient), out_channels=round_filters(320, width_coefficient),
                                  layer=round_repeats(1, depth_coefficient), stride=1, expansion_factor=6, k=3,
                                  drop_connect_rate=drop_connect_rate)
        
        self.conv2 = Conv2D(filters=round_filters(1280, width_coefficient), kernel_size=(1,1), strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()
        self.pool = GlobalAveragePooling2D()
        self.dropout = Dropout(rate=dropout_rate)
        self.fc = Dense(units=NUM_CLASSES, activation=softmax)
        
    def call(self, inputs, training=None, mask=None):
        print("EFFICIENTNET CLASS __CALL__")
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.swish(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.swish(x)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        x = self.fc(x)
        
        return x

'''
@function: efficientNet()
    - This function is a callable function from other python source code
      to get the model specification and get ready to train
    - Only works as EfficientNet B0
'''
def efficientNet():
    model = EfficientNet(1.0, 1.0, 224, 0.2)
    
    return model