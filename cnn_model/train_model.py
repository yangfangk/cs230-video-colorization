import numpy as np
import argparse
import os

from model import cnn_model

import keras.backend as K
K.set_image_data_format('channels_last')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', required=True,
    help="Directory containing the train, dev, and test .npy examples.")


"""
Used to weight the squared pixel differences between the generated frame and
the true frame and the difference between the generated frame and the previous
generated frame.
For example, if FRAME_DIFF_BETA = 0.8, then
lossFunc = (mse_of_pixel_diff_of_current_frame_to_true_frame * 0.8) +
(mse_of_squared_diff_of_current_frame_to previous_frame * 0.2)
"""
FRAME_DIFF_BETA = 0.8

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# Builds a custom Keras loss function based on the previous frame.
# y_true, y_pred shape: (#frames, 256, 256, 3)
def frame_loss(y_true, y_pred):
    num_frames = K.int_shape(y_pred)[0]

    # MSE between the true frame and the generated frame
    true_mse = mean_squared_error(y_true, y_pred)


    first_pred_frame = K.slice(y_pred, start=0, size=1)
    remaining_pred_frames = K.slice(y_pred, start=0, size=(num_frames - 1))

    prev_y_pred = K.concatenate([first_pred_frame, remaining_pred_frames])

    # MSE between the current generated frame and the prev generated frame
    prev_mse = mean_squared_error(y_pred, prev_y_pred)

    return FRAME_DIFF_BETA * true_mse + (1-FRAME_DIFF_BETA) * prev_mse


if __name__ == '__main__':
    model = cnn_model()

    model.compile(loss=frame_loss, optimizer='adam')
    print('success')

    """
    args = parser.parse_args()
    train_dir = os.path.join(args.dataset_dir, 'train')
    dev_dir = os.path.join(args.dataset_dir, 'dev')
    test_dir = os.path.join(args.dataset_dir, 'test')

    train_example_files = os.listdir(train_dir)
    train_example_files = [os.path.join(train_dir, f) for f in train_example_files if f.endswith('.npy')]

    print (train_example_files[0], train_example_files[-1])

    for train_example_file in train_example_files:
        train_example = np.load(train_example_file)
        print("train_example.shape:", train_example.shape)

        raise
    """

    # TODO: Compile model
    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer=’adam’, metrics=[“accuracy”])

    # TODO: Train/fit model
    # model.fit(x = X_train, y = Y_train, epochs = 20, batch_size = 16)
