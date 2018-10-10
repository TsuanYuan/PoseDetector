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
import pickle

def collage_images(images, w_count, h_count, norm_shape=(128,256)):
    assert len(images) <= w_count*h_count
    canvas = numpy.zeros((norm_shape[1]*h_count, norm_shape[0]*w_count, 3), dtype=numpy.uint8)
    for i, image in enumerate(images):
        im = cv2.resize(image, norm_shape)
        row_id = int(i/w_count)
        col_id = i%w_count
        canvas[row_id*norm_shape[1]:(row_id+1)*norm_shape[1],col_id*norm_shape[0]:(col_id+1)*norm_shape[0], :] = im
    return canvas


def decode_keypoints_on_collages(keypoints_on_one_collage, w_count, max_count, norm_shape=(128, 256)):
    # keypoints_on_one_collage = numpy.squeeze(keypoints_on_one_collage)
    mean_xy = numpy.mean(keypoints_on_one_collage, axis=1)[:, :2].astype(int)
    crop_idx = mean_xy[:, 1]/norm_shape[1]*w_count + mean_xy[:, 0]/norm_shape[0]
    keypoints = collections.defaultdict(list)
    n = keypoints_on_one_collage.shape[0]
    for i in range(n):
        col = crop_idx[i]%w_count
        row = crop_idx[i]/w_count
        if crop_idx[i]+1 > max_count:
            continue
        kp = numpy.squeeze(keypoints_on_one_collage[i, :])-numpy.array([norm_shape[0]*col, norm_shape[1]*row,0,0])
        keypoints[crop_idx[i]].append(kp / [norm_shape[0], norm_shape[1],1,1])
    return keypoints


def get_key_points_in_split(model, split_data, data_folder, keys, batch_size=8, w_count=8, h_count=4, norm_shape=(128, 256)):
    keypoints = {}
    for m, key in enumerate(keys):
        line = split_data[key]
        images = []
        image_counts = {}
        collages = []
        count = 0
        keypoints_decode = collections.defaultdict(list)
        for i, p in enumerate(line):
            file_name, offset = p
            data_file = os.path.join(data_folder,file_name)
            if not os.path.isfile(data_file):
                print('fail to load data file {}'.format(data_file))
                continue
            image = data_binary.read_one_image(data_file, offset)
            image = cv2.resize(image, norm_shape)
            images.append(image)
            if len(images) == w_count*h_count or i==len(line)-1:
                collage = collage_images(images, w_count, h_count)
                collages.append(collage)
                image_counts[len(collages)-1] = len(images)
                images = []
            # put w_count*h_count crops in one collage and detect keypoints. decode the keypoints to the xy in each crop afterwards
            # ignore the tail crops less than batch_size
            if len(collages) >= batch_size or i==len(line)-1:
                if len(collages)<batch_size:
                    collages = collages + [numpy.zeros(collages[0].shape, dtype=numpy.uint8) for _ in range(batch_size-len(collages))]
                keypoints_on_collages = maskrcnn.detect_keypoints_images(model, numpy.array(collages))
                batch_len = len(collages)

                for k in range(batch_len):
                    keypoints_on_one_page = keypoints_on_collages[k]
                    keypoints_row = decode_keypoints_on_collages(keypoints_on_one_page, w_count, image_counts[k], norm_shape)
                    # j is the id of a patch in current one collage
                    for j in keypoints_row:
                        # global id = k*w_count*h_count+j
                        keypoints_decode[k*w_count*h_count+j]=keypoints_row[j]
                count += w_count * h_count * batch_size
                collages = []
                images = []
        #keypoints_batch = model.compute_features_on_batch(numpy.array(images))
        keypoints[key] = keypoints_decode
        if (m+1) % 20 == 0:
            print("finished computing keypoints of pid/track_id {} out of {}".format(str(m+1), str(len(split_data))))
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
    ap.add_argument("--last_line", type=int, help="last line to process", default=30000000)
    ap.add_argument("--sample_size", type=int, help="num per track", default=8)
    ap.add_argument("--pid_dict_file", type=str, help="temp file of pids in pickle", default='/tmp/pid_dict.pkl')
    args = ap.parse_args()

    if not os.path.isfile(args.pid_dict_file):
        pid_data = load_list_to_pid(args.index_file, args.data_folder)
        with open(args.pid_dict_file, 'wb') as fp:
            pickle.dump(pid_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print('pid dict saved to {}'.format(args.pid_dict_file))
    else:
        print('pid dict being loaded from {}'.format(args.pid_dict_file))
        with open(args.pid_dict_file, 'rb') as fp:
            pid_data = pickle.load(fp)
    pids = pid_data.keys()

    inference_config = dependency.InferenceConfig()
    inference_config.GPU_COUNT = args.gpu_count
    inference_config.BATCH_SIZE = args.batch_size
    print("detect keypoints with {} gpus and batch size of {}".format(str(args.gpu_count), str(args.batch_size)))
    print("detect keypoints of {} to {} of total {}".format(str(args.start_line), str(args.last_line), str(len(pids))))
    model = maskrcnn.load_model(args.model_path, args.output_folder, inference_config)

    pids_split = pids[args.start_line:args.last_line]

    keypoints = get_key_points_in_split(model, pid_data, args.data_folder, pids_split, batch_size=args.batch_size)
    output_file = os.path.join(args.output_folder, str(args.start_line)+'_'+str(args.last_line)+'_keypoints.pkl')
    if os.path.isdir(args.output_folder) == False:
        os.makedirs(args.output_folder)
    with open(output_file, 'wb') as fp:
        pickle.dump(keypoints,fp, pickle.HIGHEST_PROTOCOL)
    print("keypoints results dumped to {}".format(output_file))