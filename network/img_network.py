# coding=utf-8
import torch.nn as nn
from torchvision import models
from network.util import *
import torch
res_dict = {"resnet18": models.resnet18, "resnet50": models.resnet50}

class ResBase(nn.Module):
    def __init__(self, res_name, pm):
        super(ResBase, self).__init__()
        if '50' in res_name:
            if pm == 'rsp':
                print("RSP model used!")
                model_resnet = res_dict[res_name](pretrained=False, num_classes=51)
                checkpoint = torch.load('/data/sihan.zhu/.cache/torch/checkpoints/rsp-resnet-50-ckpt.pth')
                pretrained_dict = checkpoint['model']
                model_resnet.load_state_dict(pretrained_dict)
            elif pm == 'imp':
                print("IMP model used!")
                model_resnet = res_dict[res_name](pretrained=True)
        elif '18' in res_name:
            print("IMP model used!")
            model_resnet = res_dict[res_name](pretrained=True)

        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
