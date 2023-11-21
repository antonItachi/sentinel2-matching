# Image matching with Sentinel2 data

## Overview
Over the course of our project, we've engaged in the development and testing of various feature matching algorithms for image processing. The focus was on implementing and optimizing classical computer vision techniques such as SIFT, ORB, and AKAZE, including the integration of RANSAC for geometric verification.

## Challenges
We tailored these algorithms to handle specific challenges posed by image datasets, such as changes as changing season, datetime and lighting conditions.

## Achievements
The project required iterative debugging and fine-tuning of parameters to improve the matching performance.

## D2-Net Usage Guide

Clone d2-net repository.

```bash
git clone https://github.com/mihaidusmanu/d2-net.git
```
### Pre-requisites
Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV
- NumPy
- PyTorch
- h5py
- imagesize
- scipy

### Downloading Weights

Download the pre-trained weights using the following commands:
```bash
wget https://dsmn.ml/files/d2-net/d2_ots.pth -O models/d2_ots.pth
wget https://dsmn.ml/files/d2-net/d2_tf.pth -O models/d2_tf.pth
wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O models/d2_tf_no_phototourism.pth
```
### Extracting Features
Run the feature extraction script in the terminal:
```bash
python extract_features.py --image_list_file "<path_to_image_list.txt>" --model_file "<path_to_weights_file>"
```
This will generate .npz files containing the features.

#### And then using this npz files, visualize keypoints matching.
