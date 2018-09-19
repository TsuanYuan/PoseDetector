"""
detector keypoints in one image
Quan Yuan
2018-09-16
"""

import argparse
import utils.maskrcnn_detector as maskrcnn


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("image_file", type=str, help="path to input image file")
    ap.add_argument("model_file", type=str, help="path to model file")
    ap.add_argument("output_folder", type=str, help="path to output folder")
    args = ap.parse_args()

    results = maskrcnn.detect_one_image(args.model_file, args.image_file)
    pass
