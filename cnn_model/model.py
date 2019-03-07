import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, Concatenate, BatchNormalization
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')

"""
import pydot
from IPython.display import SVG
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
"""

# Define image shape
IMAGE_SHAPE = (256, 256, 3)

"""
Creates a AlexNet Keras model with the following inputs (images preprocessed
into (256 x 256 x 3) images and then flattened into npy arrays):
    1. cur_colorized_frame: current colorized frame
    2. prev_colorized_frame: previous colorized frame
"""
def model():
    # Define inputs
    cur_colorized_frame = Input(IMAGE_SHAPE)
    prev_colorized_frame = Input(IMAGE_SHAPE)

    # Need to specify input_shape=(...) in first layer of network
    X = Concatenate(axis=-1)([cur_colorized_frame, prev_colorized_frame, ])

    # 1st Convolutional Layer
    X = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid') (X)
    X = Activation('relu') (X)
    # Avg Pooling
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid') (X)

    # 2nd Convolutional Layer
    X = Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)
    # Avg Pooling
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid') (X)

    # 3rd Convolutional Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)

    # 4th Convolutional Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)

    # 5th Convolutional Layer
    X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)
    # Avg Pooling
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid') (X)

    # TODO: Upsample output X to IMAGE_SIZE
  
    # Create model
    model = Model(inputs=(cur_colorized_frame, prev_colorized_frame,), outputs=X)

    model.summary()

    return model

    """
    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(17))
    model.add(Activation('softmax'))
        
    model.summary()

    # (4) Compile 
    model.compile(loss='categorical_crossentropy', optimizer='adam',\
     metrics=['accuracy'])

    # (5) Train
    model.fit(x, y, batch_size=64, epochs=1, verbose=1, \
    validation_split=0.2, shuffle=True)
    """

if __name__ == '__main__':
    model()
