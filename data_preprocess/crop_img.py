import os
import argparse
from collections import defaultdict
import logging
import pickle
import json

import numpy as np
import torch
from torch import nn
from scipy.io import loadmat

import dlib
import cv2



def compute_aspect_preserved_bbox(bbox, increase_area, h, w):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top

    width_increase = max(
        increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width)
    )
    height_increase = max(
        increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height)
    )

    left_t = int(left - width_increase * width)
    top_t = int(top - height_increase * height)
    right_t = int(right + width_increase * width)
    bot_t = int(bot + height_increase * height)

    left_oob = -min(0, left_t)
    right_oob = right - min(right_t, w)
    top_oob = -min(0, top_t)
    bot_oob = bot - min(bot_t, h)

    if max(left_oob, right_oob, top_oob, bot_oob) > 0:
        max_w = max(left_oob, right_oob)
        max_h = max(top_oob, bot_oob)
        if max_w > max_h:
            return left_t + max_w, top_t + max_w, right_t - max_w, bot_t - max_w
        else:
            return left_t + max_h, top_t + max_h, right_t - max_h, bot_t - max_h

    else:
        return (left_t, top_t, right_t, bot_t)

def crop_src_image(src_img, save_img, increase_ratio, detector=None):
    if detector is None:
        detector = dlib.get_frontal_face_detector()

    img = cv2.imread(src_img)
    faces = detector(img, 0)
    h, width, _ = img.shape
    if len(faces) > 0:
        bbox = [faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()]
        l = bbox[3] - bbox[1]
        bbox[1] = bbox[1] - l * 0.1
        bbox[3] = bbox[3] - l * 0.1
        bbox[1] = max(0, bbox[1])
        bbox[3] = min(h, bbox[3])
        bbox = compute_aspect_preserved_bbox(
            tuple(bbox), increase_ratio, img.shape[0], img.shape[1]
        )
        img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(save_img, img)
    else:
        raise ValueError("No face detected in the input image")
        # img = cv2.resize(img, (256, 256))
        # cv2.imwrite(save_img, img)

crop_src_image('source_img.jpg', 'save_img.jpg', 0.8)