"""
detector keypoints in one image
Quan Yuan
2018-09-16
"""

import argparse
import detector.maskrcnn_detector as maskrcnn
import detector.misc.utils as utils
import detector.misc.dependency as dependency
import detector.models.parallel_model as parallel_model
import cv2
import numpy

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("image_file", type=str, help="path to input image file")
    ap.add_argument("model_file", type=str, help="path to model file")
    ap.add_argument("output_folder", type=str, help="path to output folder")
    ap.add_argument("--gpu_count", type=int, default=0, help="how many gpu to use")
    args = ap.parse_args()

    batch_size = 4

    inference_config = dependency.InferenceConfig()
    inference_config.GPU_COUNT = args.gpu_count
    inference_config.BATCH_SIZE = batch_size
    model = maskrcnn.load_model(args.model_file, args.output_folder, inference_config)
    print("model path = {}".format(args.model_file))
    # if args.gpu_count > 0:
    #     model = parallel_model.ParallelModel(model, gpu_count=args.gpu_count)
    image_rgb = utils.read_one_image(args.image_file)
    results = maskrcnn.detect_keypoints_images(model, [image_rgb]*batch_size)
    n = results.shape[0]
    image_result = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for k in range(n):
        result = results[k, :]
        image_result = utils.plot_key_points(image_result, result[0, :, 0], result[0, :, 1])
    cv2.imshow('w', image_result)
    cv2.waitKey()
    pass
