import torch
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
import os
import normal_para as normalize
model_dir = '/home/stefan/PycharmProjects/parkingLot_SPP/model/'
device = gl.get_default_device()
loader = transforms.Compose([transforms.Resize((150, 150), interpolation=Image.BICUBIC),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.40, 0.39, 0.34), std=(0.19, 0.19, 0.20))])
unloader = transforms.ToPILImage()
def tensor_to_numpy(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

if __name__ == '__main__':
    testloader = dts.ImageFolder(root='/home/stefan/parkingLot/test/', transform=loader)
    testset = DataLoader(dataset=testloader, batch_size=1, shuffle=True)
    testset = gl.DeviceDataLoader(testset, device)

    # load model
    net = model.Net().to(device)
    checkpoint = torch.load(model_dir + 'sppNet_5_best.pkl')
    net.load_state_dict(checkpoint['state_dict'])
    criterion = nn.CrossEntropyLoss().to(device)
    net.to(device)
    net.eval()

    with torch.no_grad():
        val_correct = 0
        val_loss = 0.
        val_num = 0
        for i, (img, labels) in enumerate(testset, 0):
            max_output_value = []
            result = []
            # print(slotID, image.size)
            image = img.cpu().clone()
            image = image.squeeze(0)
            image = unloader(image)
            image = np.array(image)
            # print(img, image)
            image = cv2.resize(image, (150, 150), interpolation=Image.BICUBIC)
            box1 = image[0:90, 60:150]  # cropping coordinates: [y0: y1, x0: x1]
            box2 = image[30:120, 30:120]
            box3 = image[60:150, 0:90]
            box1 = Image.fromarray(box1)
            box2 = Image.fromarray(box2)
            box3 = Image.fromarray(box3)
            # resize_image = Image.fromarray(image)
            # resize_tensor = loader(resize_image).unsqueeze(0).to(device, torch.float)
            box1_tensor = loader(box1).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box2_tensor = loader(box2).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box3_tensor = loader(box3).unsqueeze(0).to(device, torch.float)  # the first dimension  is for batch size
            box_tensors = [box1_tensor, box2_tensor, box3_tensor]
            # print(img.shape)
            #
            # box1 = img.[:1, 3, 0:60, 60:150]  # cropping coordinates: [y0: y1, x0: x1]
            # box2 = img[:1, 3, 30:120, 30:120]
            # box3 = img[:1, 3, 60:150, 0:90]
            # box_tensors = [box1, box2, box3]
            pred = 0
            for tensor in box_tensors:
                outputs = net(tensor)
                # print(outputs)
                val_pred = outputs[0, 1]
                print(outputs)
                if val_pred > 0.5:
                    pred = 1
            if pred == 1:
                result.append(1)
            else:
                result.append(0)
            ori_tensor = net(img)
            ori_pred = torch.max(ori_tensor, 1)[1]

            # resize_tensor = net(resize_tensor)
            # resize_pred = torch.max(resize_tensor, 1)[1]
            result_tensor = torch.from_numpy(np.array(result)).to(device)
            correct = torch.eq(result_tensor, labels).float().sum().item()
            # print(ori_tensor)
            print(ori_pred, result_tensor, labels, '\n')
            val_correct += correct
            val_num += img.size(0)
            # if i > 10:
            #     break


        print(val_correct, val_num)
        testing_loss = val_loss / val_num
        val_acc = val_correct / val_num
        print('[overcast] val_loss : %.5f ,val_acc : %.5f' % (testing_loss, val_acc))

