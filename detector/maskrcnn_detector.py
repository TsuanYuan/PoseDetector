"""
detector pose with mask rcnn
Quan Yuan
2018-09-15
"""

from .models import mask_rcnn as modellib
from .misc import dependency as dependency
import numpy


def load_model(model_path, log_dir):
    inference_config = dependency.InferenceConfig()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=log_dir)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model


def detect_keypoints_one_image(model, image_rgb):

    results = model.detect_keypoint([image_rgb], verbose=0)
    key_points = numpy.array([result['keypoints'] for result in results])  # for one image
    return key_points
    #return r['rois'], r['keypoints'], r['masks'], r['class_ids'], r['scores']
