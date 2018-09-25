"""
detector keypoints in one image
Quan Yuan
2018-09-16
"""

import argparse
from data_utils import data_binary
import detector.maskrcnn_detector as maskrcnn
import detector.misc.utils as utils
import detector.misc.dependency as dependency
import detector.models.parallel_model as parallel_model
import cv2
import numpy
import glob, os
import detect_binary
import collections
import pickle


def process_one_folder(model, folder, norm_shape=(128,256), w_count=8, h_count=4):
    jpgs = glob.glob(os.path.join(folder, '*.jpg'))
    collages = []
    images, jpg_files, jpg_files_batch = [],[],[]
    keypoints_decode = collections.defaultdict(list)
    count = 0
    keypoints = {}
    for i, jpg_file in enumerate(jpgs):
        im_rgb = data_binary.load_rgb_image(jpg_file)
        im_rgb = cv2.resize(im_rgb, norm_shape)
        images.append(im_rgb)
        jpg_files.append(jpg_file)
        if len(images) == w_count * h_count or i == len(jpgs) - 1:
            collage = detect_binary.collage_images(images, w_count, h_count)
            collages.append(collage)
            pre_images = images
            images = []
        if len(collages) >= batch_size or i==len(jpgs)-1:
            if len(collages)<batch_size:
                collages = collages + [numpy.zeros(collages[0].shape, dtype=numpy.uint8) for _ in range(batch_size-len(collages))]
            keypoints_on_collages = maskrcnn.detect_keypoints_images(model, numpy.array(collages))
            batch_len = len(collages)

            for k in range(batch_len):
                keypoints_on_one_page = keypoints_on_collages[k]
                keypoints_row = detect_binary.decode_keypoints_on_collages(keypoints_on_one_page, w_count, h_count, norm_shape)
                # j is the id of a patch in current one collage
                for j in keypoints_row:
                    # global id = k*w_count*h_count+j
                    keypoints_decode[k*w_count*h_count+j]=keypoints_row[j]
                    keypoints[jpg_files[k*w_count*h_count+j]] = keypoints_row[j]
            count += w_count * h_count * batch_size
            collages = []
            images = []
            jpg_files = []
        #keypoints_batch = model.compute_features_on_batch(numpy.array(images))
    kp_file = os.path.join(folder, 'keypoints.pkl')
    with open(kp_file, 'wb') as fp:
        pickle.dump(keypoints, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("keypoints results dumped to {}".format(kp_file))


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str, help="path to input image folder")
    ap.add_argument("model_file", type=str, help="path to model file")
    ap.add_argument("--gpu_count", type=int, default=0, help="how many gpu to use")
    args = ap.parse_args()

    batch_size = 1

    inference_config = dependency.InferenceConfig()
    inference_config.GPU_COUNT = args.gpu_count
    inference_config.BATCH_SIZE = batch_size

    model = maskrcnn.load_model(args.model_file, '/tmp/log_model/', inference_config)
    print("model path = {}".format(args.model_file))
    sub_folders = next(os.walk(args.folder))[1]  # [x[0] for x in os.walk(folder)]
    tps = []
    for sub_folder in sub_folders:
        sub_folder_full = os.path.join(args.folder, sub_folder)
        process_one_folder(model, sub_folder_full)