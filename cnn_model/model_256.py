import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D, Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.layers import Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')

# Define image shape
IMAGE_SHAPE = (256, 256, 3)

# TODO: add BatchNormalization layers and/or switch to MaxPooling2D layers
"""
Creates a AlexNet Keras model with the following inputs (images preprocessed
into (256 x 256 x 3) images and then flattened into npy arrays):
    1. cur_colorized_frame: current colorized frame
    2. prev_colorized_frame: previous colorized frame
"""
def cnn_model():
    # Define inputs
    cur_colorized_frame = Input(IMAGE_SHAPE)
    prev_colorized_frame = Input(IMAGE_SHAPE)

    X = Concatenate(axis=-1)([cur_colorized_frame, prev_colorized_frame, ])

    # 1st Convolutional Layer
    X = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid') (X)
    X = Activation('relu') (X)
    X = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='valid') (X)
    X = BatchNormalization()(X)

    # 2nd Convolutional Layer
    X = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same') (X)
    X = Activation('relu') (X)
    X = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='valid') (X)
    X = BatchNormalization()(X)

    # 3rd Convolutional Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same') (X)
    X = Activation('relu') (X)
    X = BatchNormalization()(X)

    # 4th Convolutional Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same') (X)
    X = Activation('relu') (X)
    X = BatchNormalization()(X)

    # 5th Convolutional Layer
    X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same') (X)
    X = Activation('relu') (X)
    X = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='valid') (X)
    X = BatchNormalization()(X)
    
    # Upsample to output size.

    # 1st Upsample Layer
    X = Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)
    X = UpSampling2D(size=(2, 2)) (X)
    X = BatchNormalization()(X)
    
    # 2nd Upsample Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)
    X = UpSampling2D(size=(2, 2)) (X)
    X = BatchNormalization()(X)
    
    # 3rd Upsample Layer
    X = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)
    X = UpSampling2D(size=(2, 2)) (X)

    # 4th Upsample Layer
    X = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid') (X)
    X = Activation('relu') (X)
    X = UpSampling2D(size=(2, 2)) (X)
    X = BatchNormalization()(X)

    # 5th Upsample Layer
    X = Conv2D(filters=96, kernel_size=(5,5), strides=(1,1), padding='same') (X)
    X = Activation('relu') (X)
    X = UpSampling2D(size=(2, 2)) (X)
    X = BatchNormalization()(X)

    # 6th Upsample Layer
    X = Conv2D(filters=3, kernel_size=(5,5), strides=(1,1), padding='same') (X)
    X = Activation('relu') (X)
    X = UpSampling2D(size=(2, 2)) (X)

    # Create model
    model = Model(inputs=(cur_colorized_frame, prev_colorized_frame,), outputs=X)

    return model

