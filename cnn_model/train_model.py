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
import cv2

from model_128 import cnn_model
from model_128 import crop_video

import keras.backend as K
K.set_image_data_format('channels_last')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', required=True,
    help="Directory containing the train, dev, and test .npy examples.")
parser.add_argument('--model_save_dir', required=True,
    help="Directory to save the checkpointed model states to.")

parser.add_argument('--dev_file', default=None,
    help="Path to a .npy example file to use as colorization benchmark during training.")
parser.add_argument('--dev_output_dir', default=None,
    help="Directory to output video colorizations of dev file to.")

# MSE loss function as defined by Keras source code
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# Builds a custom Keras loss function based on the previous frame.
# y_true, y_pred shape: (#frames, 256, 256, 3)
def frame_loss(y_true, y_pred):
    # Take the center 128x128x3 crop per frame.
    y_true = crop_video(y_true)
    y_pred = crop_video(y_pred)

    # MSE between the true frame and the generated frame
    true_mse = mean_squared_error(y_true, y_pred)

    # Create matrix of previous frames
    first_pred_frame = y_pred[0:1]
    remaining_pred_frames = y_pred[:-1]
    prev_y_pred = K.concatenate([first_pred_frame, remaining_pred_frames], axis=0)
    assert(K.int_shape(y_pred) == K.int_shape(prev_y_pred))

    # MSE between the current generated frame and the prev generated frame
    prev_mse = mean_squared_error(y_pred, prev_y_pred)
    assert(K.int_shape(true_mse) == K.int_shape(prev_mse))

    # Calculate weighted loss
    weighted_loss = FRAME_DIFF_BETA * true_mse + (1-FRAME_DIFF_BETA) * prev_mse
    return weighted_loss

def colorize_video (model, i, output_dir, dev_file):
    example = np.load(dev_file)
    prev_c_frames, cur_c_frames, true_c_frames = example
    
    # Take the center 128x128x3 crop per frame.
    prev_c_frames = crop_video(prev_c_frames)
    cur_c_frames = crop_video(cur_c_frames)
    true_c_frames = crop_video(true_c_frames)

    pred_frames = model.predict(x=[cur_c_frames, prev_c_frames]).astype(np.uint8)
    print ('Epoch{} pred_frame sum:'.format(i), np.sum(pred_frames))
    
    # TODO: use true_c_frames for evaluation if desired
    fourcc = cv2.VideoWriter_fourcc(*'mp4v');
    fps = 30.0
    new_width, new_height = 256, 256

    output_path = os.path.join(output_dir, 'dev_epoch{}.mp4'.format(i))
    #print ('output_path:', output_path)

    # Recolorized output video
    color_out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (new_width, new_height),
        isColor=True
    )

    for frame in pred_frames:
        # convert RGB to BGR convention
        frame = frame[:,:,::-1]

        color_out.write(frame)

    color_out.release()


if __name__ == '__main__':
    model = cnn_model()
    model.summary()
    args = parser.parse_args()

    # Check if should colorize a video every epoch as benchmark
    if args.dev_file is None or args.dev_output_dir is None:
        colorize_benchmark = False
    else:
        colorize_benchmark = True

    # Check if model save dir exists
    if not os.path.exists(args.model_save_dir):
        os.mkdir(args.model_save_dir)
    else:
        print("Warning: output dir {} already exists".format(args.model_save_dir))

    # Check if dev output dir exists
    if colorize_benchmark:
        if not os.path.exists(args.dev_output_dir):
            os.mkdir(args.dev_output_dir)
        else:
            print("Warning: output dir {} already exists".format(args.dev_output_dir))

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

            # Take the center 128x128x3 crop per frame.
            prev_c_frames = crop_video(prev_c_frames)
            cur_c_frames = crop_video(cur_c_frames)
            true_c_frames = crop_video(true_c_frames)

            # Train model on current example
            model.fit(
                x=[cur_c_frames, prev_c_frames],
                y=true_c_frames,
                epochs=NUM_EPOCHS_PER_EXAMPLE,
                batch_size=BATCH_SIZE
            )
        
        if colorize_benchmark:
            colorize_video (model, i, args.dev_output_dir, args.dev_file)

        # Save model
        if i == (NUM_TOTAL_TRAINING_EPOCHS-1):
            model.save(os.path.join(args.model_save_dir, "trained_model.h5"))
        else:
            model.save(
                os.path.join(args.model_save_dir, "train_epoch_{}.h5".format(i))
            )
