import argparse
import os
import numpy as np

import cv2

from keras.models import load_model
from train_model import frame_loss

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', required=True,
    help="Directory containing the .npy arrays of videos to colorize.")
parser.add_argument('--model', required=True,
    help="Path to the trained model.")
parser.add_argument('--output_dir', required=True,
    help="Directory to save the colorized .mp4 videos to.")

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Load model
    model = load_model(args.model, custom_objects={'frame_loss': frame_loss})
    model.compile(loss=frame_loss, optimizer='adam')

    # Load inputs
    example_files = os.listdir(args.input_dir)
    example_files = [
        os.path.join(args.input_dir, f) for f in example_files if f.endswith('.npy')
    ]

    for example_file in example_files:
        example = np.load(example_file)
        prev_c_frames, cur_c_frames, true_c_frames = example

        pred_frames = model.predict(x=[cur_c_frames, prev_c_frames]).astype(np.uint8)
        print ('pred_frame sum:', np.sum(pred_frames))
        
        # TODO: use true_c_frames for evaluation if desired

        file_id = os.path.basename(example_file).split('.')[0]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        fps = 30.0
        new_width, new_height = 256, 256

        # Recolorized output video
        color_out = cv2.VideoWriter(
            os.path.join(args.output_dir, 'color_{}.mp4'.format(file_id)),
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


