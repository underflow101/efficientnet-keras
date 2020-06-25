import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, DepthwiseConv2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input, Concatenate
from tensorflow.keras import Sequential
from tensorflow.keras.activations import softmax

input_shape=(224,224,3)

class Conv(Layer):
    def __init__(self):
        super().__init__()
        print("INIT SEQUENCE")
        self.conv1 = Conv2D(filters=32, kernel_size=(1,1), strides=1, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
    def call(self, inputs, training=None):
        print("CALLLLLLLLLLLLLLLL")
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        print("SWISH!!!!!")
        x = tf.nn.swish(x)
        return x

model = Sequential()
inputs = tf.keras.layers.Input(shape=input_shape)
#model.add(Conv2D(filters=16, kernel_size=(1,1), strides=1, padding='same', use_bias=False)(inputs))
#model.build((None, 16))

#x = Dense(8)
#x = BatchNormalization()(x)

#model.add(tf.keras.layers.Dense(8))
#model.add(BatchNormalization())
#model.add(x)
print("1111111111111111")
model.add(Conv())
print("222222222222222222")
model.build(input_shape=(None, 224, 224, 3))
print("333333333333333333333333")
model.summary()