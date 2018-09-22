"""
detect keypoints in binary files
Quan Yuan
2018-09-19
"""

import argparse
import os
import detector.maskrcnn_detector as maskrcnn
from data_utils import data_binary
import cv2
import numpy
import collections
import detector.misc.dependency as dependency


def get_key_points_in_split(model, split_data, data_folder, keys, batch_size=64):
    keypoints = {}
    for i, key in enumerate(keys):
        line = split_data[key]
        images = []
        keypoints_row = []
        for i, p in enumerate(line):
            file_name, offset = p
            data_file = os.path.join(data_folder,file_name)
            if not os.path.isfile(data_file):
                print('fail to load data file {}'.format(data_file))
                continue
            image = data_binary.read_one_image(data_file, offset)
            images.append(image)
            if len(images) >= batch_size or i==len(line)-1:
                keypoints_batch = model.compute_features_on_batch(numpy.array(images))
                keypoints_row += keypoints_batch
                images  = []

        #keypoints_batch = model.compute_features_on_batch(numpy.array(images))
        keypoints[key] = keypoints_row
        if (i+1) % 20 == 0:
            print("finished computing descriptor of pid/track_id {} out of {}".format(str(i+1), str(len(split_data))))
    return keypoints

def load_list_to_pid(list_file, data_folder, path_tail_len=2):
    pid_index = collections.defaultdict(list)
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            label = int(line.split()[0])
            groups = line.strip().split()[1:]
            num_imgs = int(len(groups) / 2)
            # if label in pid_index:
            #     print "pid label {} already in pid_index. might have a conflict!".format(str(label))
            for i in range(num_imgs):
                part_name = groups[2 * i]
                within_idx = int(groups[2 * i + 1])
                if len(data_folder) > 0:
                    path_parts = os.path.normcase(part_name).split('/')[-path_tail_len:]
                    path_tail = os.path.join(*path_parts)
                    data_file = os.path.join(data_folder, path_tail)
                else:
                    data_file = part_name
                if os.path.isfile(data_file):
                    pid_index[label].append((data_file, within_idx))
    return pid_index


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("index_file", type=str, help="path to input index file")
    ap.add_argument("data_folder", type=str, help="path to data_folder")
    ap.add_argument("model_path", type=str, help="path to model file")
    ap.add_argument("output_folder", type=str, help="path to output file")
    ap.add_argument("--gpu_count", type=int, help="count of gpu", default=1)
    ap.add_argument("--batch_size", type=int, help="batch_size in detection", default=64)
    ap.add_argument("--start_line", type=int, help="line to start", default=0)
    ap.add_argument("--last_line", type=int, help="last line to process", default=300000)
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    args = ap.parse_args()
    pid_data = load_list_to_pid(args.index_file, args.data_folder)
    pids = pid_data.keys()

    inference_config = dependency.InferenceConfig()
    inference_config.GPU_COUNT = args.gpu_count
    inference_config.BATCH_SIZE = args.batch_size
    print("detect keypoints with {} gpus and batch size of {}".format(str(args.gpu_count), str(args.batch_size)))
    model = maskrcnn.load_model(args.model_file, args.output_folder, inference_config)

    keypoints = get_key_points_in_split(model, pid_data, args.data_folder, pids, batch_size=args.batch_size)