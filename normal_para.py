import os
import re
import numpy as np
import cv2
import gpu_loader as gl
from PIL import Image

def caculate(img):
    img_h, img_w = 150, 150
    means, stds = [], []
    img_list = []
    device = gl.get_default_device()

    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))
    means.reverse()
    stds.reverse()

    return means, stds




