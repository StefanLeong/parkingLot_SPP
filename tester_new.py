import torch
import os
import torch.nn as nn
import model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import cv2
import numpy as np
import torchvision.datasets as dts
import gpu_loader as gl
import normal_para as normalize

model_dir = '/home/stefan/PycharmProjects/parkingLot_SPP/model/'
test_path = '/home/stefan/PKLot/PKLotSegmented/Rainy/'

def test_loader(path):
    i = 0
    data_dic = []
    dir_busy = path + '/busy/'
    dir_free = path + '/free/'

    imglist = os.listdir(dir_busy) + os.listdir(dir_free)
    for item in imglist:
        if i < len(os.listdir(dir_busy)):
            img_dic = {'img_path': dir_busy + item, 'label': 1}
        else:
            img_dic = {'img_path': dir_free + item, 'label': 0}
        data_dic.append(img_dic)
        i += 1
    return data_dic


if __name__ == '__main__':
    i = 0
    val_correct = 0.0
    loader = transforms.Compose([transforms.Resize((60, 60), interpolation=Image.BICUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.52, 0.52, 0.50), std=(0.27, 0.27, 0.26))])
    device = gl.get_default_device()

    net = model.Net().to(device)
    checkpoints = torch.load(model_dir + 'sppNet_5_best.pkl')
    checkpoint = checkpoints['state_dict']
    step = checkpoints['epoch']
    net.load_state_dict(checkpoint)
    net.eval()
    test_data = test_loader(test_path)
    print(len(test_data))
    for item in test_data:
        max_value = []
        pred = 0
        path, label = item['img_path'], item['label']
        image = cv2.imread(path)
        image = cv2.resize(image, (150, 150), interpolation=Image.BICUBIC)
        box1 = image[0:90, 60:150]  # cropping coordinates: [y0: y1, x0: x1]
        box2 = image[30:120, 30:120]
        box3 = image[60:150, 0:90]
        box1 = Image.fromarray(box1)
        box2 = Image.fromarray(box2)
        box3 = Image.fromarray(box3)
        box1_tensor = loader(box1).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
        box2_tensor = loader(box2).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
        box3_tensor = loader(box3).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
        ori_tensor = loader(Image.fromarray(image)).unsqueeze(0).to(device, torch.float)
        box_tensors = [box1_tensor, box2_tensor, box3_tensor]

        for tensor in box_tensors:
            output = net(tensor)
            max_value.append(output[0, 0])
        if (max_value[0] + max_value[1] + max_value[2]) / 3 > 0.3:
            pred = 1
        if pred == label:
            val_correct += 1
        i += 1

    val_acc = val_correct / len(test_data)
    print('val_acc: ', val_acc)






