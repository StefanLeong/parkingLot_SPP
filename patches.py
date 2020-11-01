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
img_dir = '/home/stefan/CNRPark_Full_Image/FULL_IMAGE_1000x750/OVERCAST/2015-11-20/camera1/'

def image_to_tensor(slotId, image, loader, device):

    means, stds = normalize.caculate(img)
    loader = transforms.Compose([transforms.Resize((90, 90), interpolation=Image.BICUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=means, std=stds)])

    image = Image.fromarray(image)
    image_tensor = loader(image).unsqueeze(0) #the first dimension  is for batch size


    return image_tensor.to(device, torch.float)
box_set = {}
if __name__ == '__main__':
    height, weight = 90, 90
    device = gl.get_default_device()
    box_tensors = []
    box_outputs = []
    loader = transforms.Compose([transforms.Resize((90, 90)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.52, 0.52, 0.51), std=(0.25, 0.25, 0.25))])
    with open('/home/stefan/CNRPark_Full_Image/camera1.csv') as csvfile:
        reader = csv.reader(csvfile)
        for index, line in enumerate(reader):
            if index == 0:
                continue
            #add into bbox_set [SlotId,X,Y,W,H]
            box_set[int(line[0])] = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]

    img_list = os.listdir(img_dir)
    for i in range(1):

        img_name = img_list[i]
        print(img_name)
        img = cv2.imread('/home/stefan/CNRPark_Full_Image/FULL_IMAGE_1000x750/OVERCAST/2015-11-20/camera1/' + img_name)

        #draw evaluated bounding box in the parking lot
        ratio = 1000 / 2592 #original image 1000*750
        font = cv2.FONT_HERSHEY_COMPLEX
        patches = {}

        # load model
        net = model.Net().to(device)
        checkpoints = torch.load(model_dir + 'sppNet_4_best.pkl')
        checkpoint = checkpoints['state_dict']
        step = checkpoints['epoch']
        net.load_state_dict(checkpoint)
        for slotID, box in box_set.items():

            box_dim = (np.asarray(box) * ratio).astype(int)
            patch = img[box_dim[1]:box_dim[1] + box_dim[3], box_dim[0]:box_dim[0] + box_dim[2]]

            crop = cv2.resize(patch, (150, 150), interpolation=Image.BICUBIC)
            box1 = crop[0:90, 60:150]  # cropping coordinates: [y0: y1, x0: x1]
            box2 = crop[30:120, 30:120]
            box3 = crop[60:150, 0:90]
            box1 = Image.fromarray(box1)
            box2 = Image.fromarray(box2)
            box3 = Image.fromarray(box3)
            patch = Image.fromarray(crop)
            patch.save('/home/stefan/PycharmProjects/parkingLot_SPP/crops/' + str(slotID) + '.jpg')
            box1.save('/home/stefan/PycharmProjects/parkingLot_SPP/crops/' + str(slotID) + '_box1.jpg')
            box2.save('/home/stefan/PycharmProjects/parkingLot_SPP/crops/' + str(slotID) + '_box2.jpg')
            box3.save('/home/stefan/PycharmProjects/parkingLot_SPP/crops/' + str(slotID) + '_box3.jpg')
            box1_tensor = loader(box1).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box2_tensor = loader(box2).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box3_tensor = loader(box3).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box_tensors = [box1_tensor, box2_tensor, box3_tensor]

            #perform detection model on the parking lot

            patch_tensor = image_to_tensor(slotID, img, loader, device)
            outputs = net(patch_tensor)
            for box in box_tensors:
                result = net(box)
                print(result[0, 1])
            print(str(slotID) + "[patch_output]: ", outputs[0, 0], '\n')

            if outputs[0, 1] > 0.5:
                output_str = 'busy'
                color = (0, 0, 255)
                print(outputs, 'busy')
            else:
                output_str = 'free'
                color = (0, 255, 0)
                print(outputs, 'free')
            output_str = str(slotID) + ': ' + output_str
            cv2.putText(patch, output_str, (0, patch.shape[1] // 2),
                        font, box_dim[2] / 100, (0, 255, 255), 2)
            # draw final output(free or busy) onto img


