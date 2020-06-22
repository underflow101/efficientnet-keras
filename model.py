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
# Convolutional Neural Networks, CVPR 2020
###############################################################################

import h5py
import os, shutil
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend
from keras.models import Model
from keras.layers import Add, Activation, Concatenate, Conv2D, Dropout
from keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D