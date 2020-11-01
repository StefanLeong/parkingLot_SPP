import csv
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import model
import  gpu_loader as gl
import torchvision.transforms as transforms
from PIL import Image
import normal_para as normalize
model_dir = '/home/stefan/PycharmProjects/parkingLot_SPP/model/'
img_dir = '/home/stefan/CNRPark_Full_Image/FULL_IMAGE_1000x750/RAINY/2015-11-21/camera9/'
def image_to_tensor(slotID, patch, loader, device):

    means, stds = normalize.caculate(patch)
    loader = transforms.Compose([transforms.Resize((90, 90), interpolation=Image.BICUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=means, std=stds)])
    image = Image.fromarray(patch)
    image.save('/home/stefan/PycharmProjects/parkingLot_SPP/patches/' + str(slotID) + '.jpg')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

box_set = {}
if __name__ == '__main__':
    loader = transforms.Compose([transforms.Resize((90, 90), interpolation=Image.BICUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.368, 0.37, 0.36), std=(0.23, 0.22, 0.22))])
    with open('/home/stefan/CNRPark_Full_Image/camera9.csv') as csvfile:
        reader = csv.reader(csvfile)
        for index, line in enumerate(reader):
            if index == 0:
                continue
            #add into bbox_set [SlotId,X,Y,W,H]
            box_set[int(line[0])] = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
    # print({k: box_set[k] for k in list(box_set)[:3]}) # print first two bounding box

    img_list = os.listdir(img_dir)
    for item in img_list:

        img_name = item
        print(img_name)
        img = cv2.imread('/home/stefan/CNRPark_Full_Image/FULL_IMAGE_1000x750/RAINY/2015-11-21/camera9/' + img_name)

        #draw evaluated bounding box in the parking lot
        ratio = 1000 / 2592 #original image 1000*750
        font = cv2.FONT_HERSHEY_COMPLEX
        # for slotID, box in box_set.items():
        #     # print(slotID, np.asarray(box) * ratio)
        #     box_dim = (np.asarray(box) * ratio).astype(int)
        #     # print(slotID, box_dim[0], box_dim[1], box_dim[2], box_dim[3])
        #     cv2.rectangle(img, (box_dim[0], box_dim[1]), (box_dim[0] + box_dim[2], box_dim[1] + box_dim[3]), (0, 255, 0), 3)
        #     #set id in the box
        #     cv2.putText(img, str(slotID), (box_dim[0]+box_dim[2]//2-box_dim[2]//4, box_dim[1]+box_dim[3]//2), font, box_dim[2] / 100, (0, 255, 255), 2)
        #     cv2.imwrite('/home/stefan/PycharmProjects/parkingLot_SPP/results/overcast/camera9/' + str(img_name), img)

        #perform detection model on the parking lot
        # load model
        device = gl.get_default_device()
        net = model.Net().to(device)
        checkpoints = torch.load(model_dir + 'sppNet_5_best.pkl')
        checkpoint = checkpoints['state_dict']
        step = checkpoints['epoch']
        net.load_state_dict(checkpoint)
        patches = {}
        for slotID, box in box_set.items():
            max_output_value = []
            box_dim = (np.asarray(box) * ratio).astype(int)
            patch = img[box_dim[1]:box_dim[1] + box_dim[3], box_dim[0]:box_dim[0] + box_dim[2]]

            image = Image.fromarray(patch)
            # print(slotID, image.size)
            image = cv2.resize(patch, (150, 150), interpolation=Image.BICUBIC)
            box1 = image[0:90, 60:150]  # cropping coordinates: [y0: y1, x0: x1]
            box2 = image[30:120, 30:120]
            box3 = image[60:150, 0:90]
            box1 = Image.fromarray(box1)
            box2 = Image.fromarray(box2)
            box3 = Image.fromarray(box3)
            # image = cv2.resize(patch, (90, 90), interpolation=Image.BICUBIC)
            # box1 = image[0:60, 60:150]  # cropping coordinates: [y0: y1, x0: x1]
            # box2 = image[30:120, 30:120]
            # box3 = image[60:150, 0:90]
            # box1 = Image.fromarray(box1)
            # box2 = Image.fromarray(box2)
            # box3 = Image.fromarray(box3)
            origin_box = Image.fromarray(patch)
            box1_tensor = loader(box1).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box2_tensor = loader(box2).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box3_tensor = loader(box3).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            patch_tensor = loader(origin_box).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box_tensors = [box1_tensor, box2_tensor, box3_tensor]

            patch_output = net(patch_tensor)
            for tensor in box_tensors:
                outputs = net(tensor)
                # print(outputs)
                max_output_value.append(outputs[0, 0])
            # print(str(slotID) + "[patch_output]: ", patch_output[0, 0], '\n')
            # print(max_output_value)
            # patches[slotID] = patch
            # patch_tensor = image_to_tensor(slotID, patch, loader, device)
            # outputs = net(patch_tensor)
            if max(max_output_value) > 0.5:
                output_str = 'busy'
                color = (0, 0, 255)
                # print(max(max_output_value), slotID)
                # print(outputs, 'busy')
            else:
                output_str = 'free'
                color = (0, 255, 0)
                # print(outputs, 'free')
            output_str = str(slotID) + ':' + output_str
            # cv2.putText(patch, output_str, (0, patch.shape[1] // 2),
            #             font, box_dim[2] / 100, (0, 255, 255), 2)
            # draw final output(free or busy) onto img
            cv2.rectangle(img, (box_dim[0], box_dim[1]), (box_dim[0] + box_dim[2], box_dim[1] + box_dim[3]), color, 3)
            # put id text at the top of the bbox
            cv2.putText(img, output_str, (box_dim[0] + box_dim[2] // 2 - box_dim[2] // 4, box_dim[1] + box_dim[3] // 2), font, box_dim[2] / 100, (0, 255, 255), 2)
            cv2.imwrite('/home/stefan/PycharmProjects/parkingLot_SPP/results/rainy/camera9/' + str(img_name), img)

