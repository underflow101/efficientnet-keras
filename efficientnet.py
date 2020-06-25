###########################################################################
# efficientnet.py
#
# Dev. Dongwon Paek
# Description: EfficientNet B0 Source Code
#              Run this code with $ python efficientnet.py on terminal
###########################################################################

import h5py
import os, shutil
import matplotlib.pyplot as plt

import keras

from model import efficientNet
from hyperparameter import NUM_CLASSES, EPOCHS, BATCH_SIZE, WIDTH, HEIGHT, CHANNELS

print("INITIALIZE OBJECT")
model = efficientNet()
print("BUILD MODEL")
model.build(input_shape=(None, HEIGHT, WIDTH, CHANNELS))
print("SUMMARY MODEL")
model.summary()

model.save('lpl.h5')

print("DONE!!!!!!!")