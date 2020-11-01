import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import model
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dts
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import gpu_loader as gl
import visdom

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
acc_best = float('-inf')
model_dir = '/home/stefan/PycharmProjects/parkingLot_SPP/model/'
size_num = [150, 60]
global_step = 0
EPOCH = 5
#---------------------准备dataset

viz = visdom.Visdom()
viz.line([0.0], [0.], win='train_loss', opts=dict(title='train_loss'))
viz.line([0.0], [0.], win='val_loss', opts=dict(title='val_loss'))
viz.line([0.0], [0.], win='train_acc', opts=dict(title='train_acc'))
viz.line([0.0], [0.], win='val_acc', opts=dict(title='val_acc'))

def loader(size):
    train_set = dts.ImageFolder(root='/home/stefan/parkingLot/train_60', transform=
                                transforms.Compose([
                                    transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.41, 0.40, 0.35), std=(0.17, 0.16, 0.17))
                                ]))
    train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)
    print('Train_size: ', len(train_loader))

    test_set = dts.ImageFolder(root='/home/stefan/parkingLot/validation_60', transform=
                                transforms.Compose([
                                    transforms.Resize((size, size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.41, 0.40, 0.35), std=(0.19, 0.17, 0.18))
                                ]))
    test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True)
    print('Test_size: ', len(test_loader))

    return train_loader, test_loader


#training
def train(model, device, train_loader, criterion, optimizer, epoch):
    global global_step
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_sum = 0
    for i, data in enumerate(train_loader, 0):
        #gte the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #zeros the para gradients

        outputs = net(inputs)
        # print(outputs., labels)
        loss = criterion(outputs, labels)

        train_pred = torch.max(outputs, 1)[1]
        correct = torch.eq(train_pred, labels).float().sum().item()
        train_correct += correct
        train_sum += inputs.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 更新参数
        # scheduler.step()
        train_loss += loss.item() #this running loss is an average value
        global_step += 1
        train_acc = train_correct / train_sum
        viz.line([train_acc], [global_step], win='train_acc', update='append')
        if i % 1000 == 999:
            print('[%d, %5d] loss : %.3f ,train_acc : %.3f' % (epoch, i + 1, train_loss / 1000, train_acc))
            viz.line([train_loss / 1000], [i], win='train_loss', update='append')
            train_loss = 0.0


def test(model, device, test_loader, criterion, optimizer, epoch):
    global global_step, acc_best
    model.eval()
    val_correct = 0
    loss = 0.0
    val_num = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss += criterion(outputs, labels).item()

        val_pred = torch.max(outputs, 1)[1]
        correct = torch.eq(val_pred, labels).float().sum().item()
        val_correct += correct
        val_num += inputs.size(0)

    testing_loss = loss / val_num
    val_acc = val_correct / val_num
    viz.line([testing_loss], [global_step], win='val_loss', update='append')
    viz.line([val_acc], [global_step], win='val_acc', update='append')

    print('[%d, %5d] loss : %.3f ,val_acc : %.3f' % (epoch, i + 1, testing_loss, val_acc))
    # viz.line([val_acc], [i], win='val_acc', update='append')
    #
    # if val_acc > acc_best:
    #     acc_best = val_acc
    #     state = {'state_dict': net.state_dict(), 'epoch': epoch}
    #     model_save = model_dir + 'sppNet_' + str(epoch) + '_best.pkl'
    #     torch.save(state, model_save)
    #     print('save the best accuracy model done!')

    state = {'state_dict': net.state_dict(), 'epoch': epoch}
    model_save = model_dir + 'sppNet_' + str(epoch) + '_best.pkl'
    torch.save(state, model_save)
    print('save the model done!')

    print('[epoch: %d] val_acc : %.3f ' % (epoch, val_acc))

if __name__ == '__main__':
    train_loader_150, test_loader_150 = loader(150)
    train_loader_60, test_loader_60 = loader(60)
    train_loaders = [train_loader_60, train_loader_150]
    test_loaders = [test_loader_60, test_loader_150]

    device = gl.get_default_device()
    net = model.Net().to(device)
    # 定义损失函数，优化方法
    # 采用cross=Entropy loss, SGD with moment
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # load model
    checkpoints = torch.load('/home/stefan/PycharmProjects/parkingLot_SPP/' + 'sppNet_8_best.pkl')
    checkpoint = checkpoints['state_dict']
    step = checkpoints['epoch']
    net.load_state_dict(checkpoint)
    print('load model done...')
    for epoch in range(1, EPOCH + 1):
        for train_loader, test_loader in zip(train_loaders, test_loaders):
            print('start training EPOCH %d ' % epoch)
            train(net, device, train_loader, criterion, optimizer, epoch)
            print('start testing EPOCH %d ' % epoch)
            test(net, device, test_loader, criterion, optimizer, epoch)

