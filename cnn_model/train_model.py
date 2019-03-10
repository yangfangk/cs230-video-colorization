#################### Training Hyperparameters ####################

# Number of epochs to train over all examples
# Note: each training epoch generates a ~110MB model checkpoint
NUM_TOTAL_TRAINING_EPOCHS = 20

# Number of epochs to train per example
NUM_EPOCHS_PER_EXAMPLE = 1

# Batch size when training each example
BATCH_SIZE = 16

"""
Used to weight the squared pixel differences between the generated
frame and the true frame and the difference between the generated
frame and the previous generated frame.
For example, if FRAME_DIFF_BETA = 0.8, then
lossFunc = (mse_of_pixel_diff_of_current_frame_to_true_frame * 0.8)
+ (mse_of_squared_diff_of_current_frame_to previous_frame * 0.2)
"""
FRAME_DIFF_BETA = 0.8

##################################################################

import numpy as np
import argparse
import os

from model import cnn_model

import keras.backend as K
K.set_image_data_format('channels_last')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', required=True,
    help="Directory containing the train, dev, and test .npy examples.")
parser.add_argument('--model_save_dir', required=True,
    help="Directory to save the checkpointed model states to.")

# MSE loss function as defined by Keras source code
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# Builds a custom Keras loss function based on the previous frame.
# y_true, y_pred shape: (#frames, 256, 256, 3)
def frame_loss(y_true, y_pred):
    # MSE between the true frame and the generated frame
    true_mse = mean_squared_error(y_true, y_pred)

    # Create matrix of previous frames
    first_pred_frame = y_pred[0:1]
    remaining_pred_frames = y_pred[1:]
    prev_y_pred = K.concatenate([first_pred_frame, remaining_pred_frames], axis=0)
    assert(K.int_shape(y_pred) == K.int_shape(prev_y_pred))

    # MSE between the current generated frame and the prev generated frame
    prev_mse = mean_squared_error(y_pred, prev_y_pred)
    assert(K.int_shape(true_mse) == K.int_shape(prev_mse))

    # Calculate weighted loss
    weighted_loss = FRAME_DIFF_BETA * true_mse + (1-FRAME_DIFF_BETA) * prev_mse
    return weighted_loss


if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    else:
        print("Warning: output dir {} already exists".format(args.model_save_dir))

    # Create model, use model.summary() to print model architecture
    model = cnn_model()
    model.compile(loss=frame_loss, optimizer='adam')

    # Load training examples
    train_dir = os.path.join(args.dataset_dir, 'train')
    train_example_files = os.listdir(train_dir)
    train_example_files = [
        os.path.join(train_dir, f) for f in train_example_files if f.endswith('.npy')
    ]

    # Train model
    for i in range(NUM_TOTAL_TRAINING_EPOCHS):
        for train_example_file in train_example_files:
            print(
                "\nTraining epoch {}/{}.".format(i+1, NUM_TOTAL_TRAINING_EPOCHS),
                "Training on \'{}\':".format(os.path.basename(train_example_file))
            )
            
            train_example = np.load(train_example_file)
            prev_c_frames, cur_c_frames, true_c_frames = train_example

            # Train model on current example
            model.fit(
                x=[cur_c_frames, prev_c_frames],
                y=true_c_frames,
                epochs=NUM_EPOCHS_PER_EXAMPLE,
                batch_size=BATCH_SIZE
            )
        
        # Save model
        if i == (NUM_TOTAL_TRAINING_EPOCHS-1):
            model.save(os.path.join(args.model_save_dir, "trained_model.h5"))
        else:
            model.save(
                os.path.join(args.model_save_dir, "train_epoch_{}.h5".format(i))
            )
