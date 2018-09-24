"""
process binary format
Quan Yuan
2018-09-19
"""
import os
import collections
import numpy
import struct
import cv2

def load_list_to_pid(list_file, data_folder, prefix, path_tail_len=2):
    pid_index = collections.defaultdict(list)
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            label = int(line.split()[0])+prefix
            groups = line.strip().split()[1:]
            num_imgs = len(groups) / 2

            for i in xrange(num_imgs):
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


def read_one_image(data_file_path, place):
    with open(data_file_path, 'rb') as f:
        f.seek(place + 4)
        name_len = struct.unpack('i', f.read(4))[0]
        f.seek(name_len, 1)
        img_len = struct.unpack('i', f.read(4))[0]
        img_bgr = cv2.imdecode(numpy.asarray(bytearray(f.read(img_len)), dtype="uint8"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img


def load_rgb_image(image_file):
    im_bgr = cv2.imread(image_file)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    return im_rgb