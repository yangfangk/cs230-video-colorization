import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')

import pydot
from IPython.display import SVG
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
%matplotlib inline

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

    X = Concatenate(axis=-1)([cur_colorized_frame, prev_colorized_frame, ])

    # 1st Convolutional Layer
    X = Conv2D(filters=96, input_shape=IMAGE_SHAPE, kernel_size=(11,11), strides=(4,4), padding=’valid’) (X)
    X = Activation(‘relu’) (X)
    # Avg Pooling
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’) (X)

    # 2nd Convolutional Layer
    X = Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding=’valid’) (X)
    X = Activation(‘relu’) (X)
    # Avg Pooling
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’) (X)

    # 3rd Convolutional Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’) (X)
    X = Activation(‘relu’) (X)

    # 4th Convolutional Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding=’valid’) (X)
    X = Activation(‘relu’) (X)

    # 5th Convolutional Layer
    X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding=’valid’) (X)
    X = Activation(‘relu’) (X)
    # Avg Pooling
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2), padding=’valid’) (X)

    # TODO: Upsample output X to IMAGE_SIZE
  
    # Create model
    model = Model(inputs=(cur_colorized_frame, prev_colorized_frame,), outputs=X)

    return model

    """
    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(256*256*3,)))
    model.add(Activation(‘relu’))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation(‘relu’))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation(‘relu’))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(17))
    model.add(Activation(‘softmax’))

    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=’adam’, metrics=[“accuracy”])
    """
