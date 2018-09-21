"""
detect keypoints in binary files
Quan Yuan
2018-09-19
"""

import argparse
import os
from detector.misc import utils
from data_utils import data_binary
import cv2
import numpy


def get_key_points_in_split(model, split_data, data_folder):
    keypoints = {}
    for i, video_track in enumerate(split_data.keys()):
        line = split_data[video_track]
        images = []
        for p in line:
            file_name, offset = p
            data_file = os.path.join(data_folder,file_name)
            if not os.path.isfile(data_file):
                print('fail to load data file {}'.format(data_file))
                continue
            image = data_binary.read_one_image(data_file, offset)
            images.append(image)
        keypoints_batch = model.compute_features_on_batch(numpy.array(images))
        keypoints[video_track] = keypoints_batch

        if (i+1) % 100 == 0:
            print("finished computing descriptor of pid/track_id {} out of {}".format(str(i+1), str(len(split_data))))
    return keypoints

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("index_file", type=str, help="path to input index file")
    ap.add_argument("data_folder", type=str, help="path to data_folder")
    ap.add_argument("model_path", type=str, help="path to model file")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--gpu_id", type=int, help="gpu_device", default=7)
    ap.add_argument("--start_line", type=int, help="line to start", default=0)
    ap.add_argument("--last_line", type=int, help="last line to process", default=300000)
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    args = ap.parse_args()

