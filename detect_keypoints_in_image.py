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
import detect_binary

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("image_file", type=str, help="path to input image file")
    ap.add_argument("model_file", type=str, help="path to model file")
    ap.add_argument("output_folder", type=str, help="path to output folder")
    ap.add_argument("--gpu_count", type=int, default=0, help="how many gpu to use")
    args = ap.parse_args()

    batch_size = 1

    inference_config = dependency.InferenceConfig()
    inference_config.GPU_COUNT = args.gpu_count
    inference_config.BATCH_SIZE = batch_size

    model = maskrcnn.load_model(args.model_file, args.output_folder, inference_config)
    print("model path = {}".format(args.model_file))
    # if args.gpu_count > 0:
    #     model = parallel_model.ParallelModel(model, gpu_count=args.gpu_count)
    image_rgb = utils.read_one_image(args.image_file)
    image_rgb2 = utils.read_one_image('/Users/quan/Documents/data/wan/00059856/ch11010_20180821203030_00004021_00012850.jpg')
    images = [image_rgb, image_rgb2, image_rgb, image_rgb2, image_rgb2, image_rgb]
    images = [cv2.resize(im, (128, 256)) for im in images]
    collage = detect_binary.collage_images(images, w_count=3, h_count=2)

    results = maskrcnn.detect_keypoints_images(model, [collage])
    results_decoded = detect_binary.decode_keypoints_on_collages(results, 3, 2)
    s = results.shape #numpy.squeeze(results).shape
    image_result = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)
    for k in range(s[0]):
        result = results[k, :]
        for j in range(s[1]):
            image_result0 = utils.plot_key_points(images[j],
                                                  results_decoded[j][0][:, 0]*images[j].shape[1],
                                                  results_decoded[j][0][:, 1]*images[j].shape[0])
            image_result2 = utils.plot_key_points(image_result, result[j, :, 0], result[j, :, 1])
            cv2.imshow('h', image_result0)
            #cv2.imshow('w', image_result2)
            cv2.waitKey()
    pass
