import torch
import torch.nn as nn
import model
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import numpy as np
import torchvision.datasets as dts
import gpu_loader as gl

model_dir = '/home/stefan/PycharmProjects/parkingLot_SPP/model/'
device = gl.get_default_device()
transforms = transforms.Compose(
    [transforms.Resize((150, 150), interpolation=Image.BICUBIC),
     transforms.ToTensor(),  #transform the images into Tensors, normalization to [0, 1]
     transforms.Normalize((0.52, 0.51, 0.50), std=(0.25, 0.25, 0.25))])

testloader = dts.ImageFolder(root='/home/stefan/PKLot/PKLotSegmented/Cloudy/', transform=transforms)
testset = DataLoader(dataset=testloader, batch_size=4, shuffle=True)

testset = gl.DeviceDataLoader(testset, device)
# load model
net = model.Net().to(device)
checkpoint = torch.load(model_dir + 'sppNet_5_best.pkl')
net.load_state_dict(checkpoint['state_dict'])
criterion = nn.CrossEntropyLoss().to(device)
net.to(device)
net.eval()
print(len(testloader))
with torch.no_grad():
    val_correct = 0
    val_loss = 0.
    val_num = 0
    for i, (img, labels) in enumerate(testset, 0):
        inputs, labels = img, labels
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        val_loss += criterion(outputs, labels).item()
        val_pred = torch.max(outputs, 1)[1] #prediction class of each box
                                            #0: the max value of every row  1: the max value of every column
                                            #[1]: the max value [1]: the index of the max value

        # print(torch.max(outputs, 1)[0])
        # print(torch.max(outputs, 1)[1])
        # print(labels)
        # print(val_pred)
        correct = torch.eq(val_pred, labels).float().sum().item()
        val_correct += correct
        val_num += inputs.size(0)
        i += 1
        # if i > 100:
        #     break
    print(val_correct, val_num)
    testing_loss = val_loss / val_num
    val_acc = val_correct / val_num
    print('[overcast] val_loss : %.5f ,val_acc : %.5f' % (testing_loss, val_acc))

