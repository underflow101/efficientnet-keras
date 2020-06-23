###############################################################################
# model.py
#
# Dev. Dongwon Paek
# Description: EfficientNet B0 model source code
#              This source code only works with efficientnet.py source code
#              This source code is different from original paper due to its
#              procedure code format
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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax

from utils import round_filters, round_repeats
from hyperparameter import NUM_CLASSES, EPOCHS, BATCH_SIZE, WIDTH, HEIGHT, CHANNELS

'''
@function: SEBlock
    - SEBlock stands for "Squeeze & Excitation" Block
    - This block reduces and expands the layer for model size
    - For reusing SEBlock, it is designed in class form
'''
def SEBlock()