import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--processed_dir',
        type=str,
        help='Directory of processed bw_ and color_ videos.',
        required=True
        )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    i = 0

    for file in os.listdir(args.processed_dir):
        # Rename each set of (bw, color) video files.
        if file.split('_')[0] == 'bw':
            filename = file[3:]

            # Rename files.
            os.rename('bw_' + filename, 'bw_{}.mp4'.format(str(i).zfill(3)))
            os.rename('color_' + filename, 'color_{}.mp4'.format(str(i).zfill(3)))
            
            i += 1

if __name__ == '__main__': 
    main() 