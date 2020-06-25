import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, Concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax

input_shape=(224,224,3)

def Convinit():
    conv1 = Conv2D(filters=32, kernel_size=(1,1), strides=1, padding='same', use_bias=False)
    bn1 = BatchNormalization()
    
    return conv1#, bn1

def Convcall(input):
    x = Conv2D(filters=32, kernel_size=(1,1), strides=1, padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = tf.nn.swish(x)
    
    return x

model = Sequential()
inputs = tf.keras.layers.Input(shape=input_shape)

print("1111111111111111")
model.add(Convinit())
model.add(BatchNormalization())
print("SWISH!!!!!!!!!!!!!")
model.add(Convcall(input=(None, 224, 224, 3)))

print("222222222222222222")
model.build(input_shape=(None, 224, 224, 3))
print("333333333333333333333333")
model.summary()