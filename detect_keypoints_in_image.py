"""
detector keypoints in one image
Quan Yuan
2018-09-16
"""

import argparse
import detector.maskrcnn_detector as maskrcnn
import detector.misc.utils as utils
import cv2

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("image_file", type=str, help="path to input image file")
    ap.add_argument("model_file", type=str, help="path to model file")
    ap.add_argument("output_folder", type=str, help="path to output folder")
    args = ap.parse_args()
    model = maskrcnn.load_model(args.model_file, args.output_folder)
    image_rgb = utils.read_one_image(args.image_file)
    results = maskrcnn.detect_keypoints_one_image(model, image_rgb)
    image_result = utils.plot_key_points(image_rgb, results[:,0], results[:,1])
    cv2.imshow('w', image_result)
    cv2.waitKey()
    pass
