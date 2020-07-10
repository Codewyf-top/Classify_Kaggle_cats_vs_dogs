import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='/home/cyh/视频/Classifer/datasets/train/')
parser.add_argument('--output', type=str, default='/home/cyh/视频/Classifer/1.flist')
args = parser.parse_args()

ext = {'.jpg', '.png', 'JPG', 'PNG'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1] in ext:
            images.append(os.path.join(root, file))

np.savetxt(args.output, images, fmt='%s')
