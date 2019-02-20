# CS230 Video Colorization
Stanford CS 230: Deep Learning (Winter 2019) Final Project

### Group members

Yang Fang, Vedi Chaudhri

### Acknowledgements

The source code for the baseline video recolorization algorithm (independent frame-by-frame image recolorization) is largely based on Gael Colas and Rafael Rafailov's [Automatic Video Colorization](https://github.com/ColasGael/Automatic-Video-Colorization) project, as well as Richard Zhang, Phillip Isola, Alexei A. Efros's [Colorful Image Colorization](https://github.com/richzhang/colorization).

## Setting up an enviroment

1. Create an VM instance (Google Cloud or AWS is used for this project), a GPU instance is not required for the baseline algorithm
2. Install anaconda3 as directed [here](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)
3. Run:
```
conda install -c conda-forge opencv
conda install -c conda-forge caffe
```
4. From the root directory, run:
```
pip install -r requirements.txt
```

## Preprocessing

For this project, we will be using the MIT Moments in Time dataset, a large dataset of short, color videos of labeled categories. In this project, we will be focusing primarily on the 'hiking' subset.

### Download dataset
1. Download and unzip the dataset:
```
wget http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_Mini.zip
unzip Moments_in_Time_Mini.zip -d data/.
```
2. Pre-process the dataset:
```
./convert_moment_dataset.sh
```

### Preprocess videos

1. Create the data directories
```
mkdir data; mkdir data/raw; mkdir data/converted;
```
2. Place videos inside 'data/raw' directory
3. Run the conversion script:
```
# For all videos inside 'data/raw' directory
python3 converter.py

# For one specific video 'filename'
python3 converter.py --inputname filename

# Convert all videos in the data/raw folder to a consistent fps and resolution
python3 converter.py --fps 30 --out_dim 640 360
```

## Running the baseline
1. Download the pretrained image colorization model:
```./models/fetch_release_models.sh```

2. Run the following command to colorize your video:
```
python3 baseline_colorize.py --filename <bw_filename> --input_dir ../data/hiking_processed/ --output_dir ../data/hiking_colorized 

# Compute and print the average pixel-match percentage with the specified (true color) video
python3 baseline_colorize.py --filename <bw_filename> --input_dir ../data/hiking_processed/ --output_dir ../data/hiking_colorized --true_path <color_file_path>
```

### Baseline results
Sample baseline colorization results for 5 videos (selected to encompass a variety of envrioments, such as the forest, desert, ocean, etc.) are available in the 'hiking_baseline' directory. Their associated pixel match percentages (averaged across all frames):
1. 25.056
2. 40.732
3. 23.562
4. 40.669
5. 40.767
