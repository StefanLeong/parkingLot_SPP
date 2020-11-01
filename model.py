import torch
import torch.nn as nn
import torch.nn.functional as tnf
import spp_layer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.output_num = [3, 2, 1]
        self.conv1 = nn.Conv2d(3, 96, (3, 3))
        self.conv2 = nn.Conv2d(96, 256, (3, 3))
        self.conv3 = nn.Conv2d(256, 128, (3, 3))
        self.conv4 = nn.Conv2d(128, 128, (3, 3))

        self.fc1 = nn.Linear(128 * sum([i * i for i in self.output_num]), 512)
        self.fc2 = nn.Linear(512, 2) ##2 is the classes num
    def forward(self, x):
        x = tnf.max_pool2d(tnf.relu(self.conv1(x)), (2, 2))
        x = tnf.max_pool2d(tnf.relu(self.conv2(x)), (2, 2))
        x = tnf.max_pool2d(tnf.relu(self.conv3(x)), (2, 2))
        x = tnf.relu(self.conv4(x))
        # print(x.size(0), x.size(1), x.size(2), x.size(3))
        spp = spp_layer.spatial_pyramid_pool(x, x.size(0), [int(x.size(2)), int(x.size(3))], self.output_num)
        #x = x.view(-1, self.num_flat_features(x))
        x = tnf.relu(self.fc1(spp))
        x = torch.sigmoid(self.fc2(x))
        return x
        # 使用该函数计算tensor x 的总特征量
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
