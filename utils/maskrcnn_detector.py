"""
detector pose with mask rcnn
Quan Yuan
2018-09-15
"""

from .models import mask_rcnn as modellib
from .misc import dependency as dependency


def load_model(model_path, log_dir):
    inference_config = dependency.InferenceConfig()
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=log_dir)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


def detect_one_image(model, image_rgb):
    results = model.detect_keypoint([image_rgb], verbose=0)
    r = results[0]  # for one image
    return r['rois'], r['keypoints'], r['masks'], r['class_ids'], r['scores']