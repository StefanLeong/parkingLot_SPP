import os
import cv2
from PIL import Image
import numpy as np
ori_dir = '/home/stefan/parkingLot/validation/free/'
img_60_dir = '/home/stefan/parkingLot/validation_60/free/'


img_list = os.listdir(ori_dir)
print(len(img_list))
for item in img_list:
    image = cv2.imread(ori_dir + item)
    image = cv2.resize(image, (60, 60))

    cv2.imwrite(img_60_dir + item, image)

