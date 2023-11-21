import argparse

import numpy as np

import imageio

import torch
from PIL import Image
from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--image_list_file', type=str, required=True,
    help='path to a file containing a list of images to process'
)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default='../models/d2_tf.pth',
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--output_extension', type=str, default='.d2-net',
    help='extension for the output'
)
parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

args = parser.parse_args()

print(args)

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_relu=args.use_relu,
    use_cuda=use_cuda
)

# Process the file
with open(args.image_list_file, 'r') as f:
    lines = f.readlines()
for line in tqdm(lines, total=len(lines)):
    path = line.strip()

    image = Image.open(path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    original_size = image.size
    max_edge = args.max_edge
    max_sum_edges = args.max_sum_edges

    # Resize the image maintaining the aspect ratio
    ratio = min(max_edge / max(original_size), max_sum_edges / sum(original_size))
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to numpy array and preprocess for the model
    image_np = np.array(image).astype('float32')
    fact_i = original_size[1] / new_size[1]
    fact_j = original_size[0] / new_size[0]

    input_image = preprocess_image(image_np, preprocessing=args.preprocessing)
    input_tensor = torch.tensor(input_image[np.newaxis, :, :, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(input_tensor, model)
        else:
            keypoints, scores, descriptors = process_multiscale(input_tensor, model, scales=[1])

    # Adjust the keypoints to the original image size
    keypoints[:, 0] *= fact_j
    keypoints[:, 1] *= fact_i
    keypoints = keypoints[:, [1, 0, 2]]  # Swap x and y coordinates

    # Save the features
    output_filename = path + args.output_extension
    if args.output_type == 'npz':
        np.savez(output_filename, keypoints=keypoints, scores=scores, descriptors=descriptors)
    elif args.output_type == 'mat':
        scipy.io.savemat(output_filename, {'keypoints': keypoints, 'scores': scores, 'descriptors': descriptors})
    else:
        raise ValueError('Unknown output type.')

