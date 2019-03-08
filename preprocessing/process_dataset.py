"""
Converts and saves the processed .mp4 (colorized, true color) videos into
files of numpy arrays.
    - Each frame is converted to a 256x256x3 numpy array
    - Each frame example is built as an array of 
      [prev_colorized_frame, cur_colorized_frame, cur_true_frame]
    - Each example is created as an array of consecutive frame examples,
      [<frame example 1>, [<frame example 2>], ...] and saved to file
"""

import argparse
import random
import os

import numpy as np

from tqdm import tqdm

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--colorized_videos_dir', required=True,
    help="Directory containing the processed, colorized videos.")
parser.add_argument('--true_videos_dir', required=True, 
    help="Directory containing the true-colored videos.")
parser.add_argument('--output_dir', required=True, help="Output directory.")

def process_and_save(filepath, true_videos_dir, output_dir):
    # Extract the 3-digit file id
    filename = os.path.basename(filepath)
    assert (filename[-4:] == '.mp4')
    file_id = filename[:-4].split('_')[-1]

    true_filepath = os.path.join(true_videos_dir, 'color_{}.mp4'.format(file_id))

    colorized_vidcap = cv2.VideoCapture(filepath)
    true_vidcap = cv2.VideoCapture(true_filepath)

    c_success, c_frame = colorized_vidcap.read()
    c_frame = c_frame[:,:,::-1] # convert BGR to RGB convention

    prev_c_frame = c_frame

    t_success, t_frame = true_vidcap.read()
    t_frame = t_frame[:,:,::-1] # convert BGR to RGB convention

    frames = []

    # Create and save npy arrays
    while c_success:
        # Assert that colorized and true videos have the same # of frames
        assert (t_success) 

        frame_example = [prev_c_frame, c_frame, t_frame]
        frames.append(frame_example)

        prev_c_frame = c_frame

        # Read next frame(s)
        c_success, c_frame = colorized_vidcap.read()
        if not c_success:
            break
        else:
            c_frame = c_frame[:,:,::-1] # convert BGR to RGB convention

            t_success, t_frame = true_vidcap.read()
            assert (t_success)

            t_frame = t_frame[:,:,::-1] # convert BGR to RGB convention

    assert (len(frames[-1]) == 3)
    assert (len(frames[-1][0]) == 256)
    assert (len(frames[-1][0][0]) == 256)
    assert (len(frames[-1][0][0][0]) == 3)

    # sanity check to ensure that the prev_c_frame of the last frame example
    # matches the c_frame of the second to last example
    assert (np.sum (frames[-2][1]) == np.sum (frames[-1][0]))

    np.save(output_dir + "/example{}.npy".format(file_id), frames)

if __name__ == '__main__':
    args = parser.parse_args()

    # Create the output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Retrieve the colorized videos
    cv_dir = args.colorized_videos_dir
    colorized_files = os.listdir(cv_dir)
    colorized_files = [os.path.join(cv_dir, f) for f in colorized_files if f.endswith('.mp4')]

    # Deterministically shuffle the colorized files
    random.seed(1337)
    colorized_files.sort()
    random.shuffle(colorized_files)

    # Split files into 80% train, 10% dev, and 10% test sets
    train_end = int(0.8 * len(colorized_files))
    dev_end = int(0.9 * len(colorized_files))

    train_set = colorized_files[:train_end]
    dev_set = colorized_files[train_end:dev_end]
    test_set = colorized_files[dev_end:]

    assert (len(train_set) + len(dev_set) + len(test_set) == len(colorized_files))

    datasets = {'train': train_set, 'dev': dev_set, 'test': test_set}

    for dataset, filenames in datasets.items():
        output_dir = os.path.join(args.output_dir, dataset)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        else:
            print("Warning: output dir {} already exists".format(output_dir))

        print("Processing {} data:".format(dataset),
            "saving processed dataset to {}".format(output_dir))

        for filename in tqdm(filenames):
            process_and_save(filename, args.true_videos_dir, output_dir)

