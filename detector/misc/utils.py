"""
utility functions
Quan Yuan
2018-09-19
"""
import cv2


def read_one_image(image_path):
    im_bgr = cv2.imread(image_path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb


def plot_key_points(im_rgb, xs, ys, radius=4):
    color = (0, 255, 0)
    count = 0
    for x, y in zip(xs, ys):
        cv2.circle(im_rgb, (int(x), int(y)),
                   radius, color, thickness=2)
        cv2.putText(im_rgb, str(count), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), 1)
        count += 1
    return im_rgb


def save_one_image(im_rgb, im_path):
    im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_path, im_bgr)
