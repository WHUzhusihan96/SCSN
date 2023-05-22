# coding=utf-8
import torch.nn as nn
from torchvision import models
from network.util import *
import torch
res_dict = {"resnet18_scsn": models.resnet18, "resnet50_scsn": models.resnet50}
import torch.nn.functional as F


class simam_module(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1
        # print(x.shape)
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # print(x.mean(dim=[2, 3], keepdim=True).shape)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        at = self.activaton(y)
        return x * at, x * (1 - at), at


class ResSCSN_Base(nn.Module):
    def __init__(self, res_name, pm):
        super(ResSCSN_Base, self).__init__()
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
        self.SimAM = simam_module()
        if '18' in res_name:
            self.IN1 = nn.InstanceNorm2d(64, affine=True)
            self.IN2 = nn.InstanceNorm2d(128, affine=True)
            self.IN3 = nn.InstanceNorm2d(256, affine=True)
            self.fc3 = nn.Linear(256, 7)
            self.fc = nn.Linear(256, 7)
            self.bt = nn.Linear(512, 256)
        elif '50' in res_name:
            self.IN1 = nn.InstanceNorm2d(256, affine=True)
            self.IN2 = nn.InstanceNorm2d(512, affine=True)
            self.IN3 = nn.InstanceNorm2d(1024, affine=True)
            self.fc3 = nn.Linear(1024, 7)
            self.fc = nn.Linear(2048, 7)

    def forward(self, x):
        end_points = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1
        x_style_1a, _, _ = self.SimAM(x_style_1)
        # x_1 = x_IN_1
        x_1 = x_IN_1 + x_style_1a

        x_2 = self.layer2(x_1)
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2a, _, _ = self.SimAM(x_style_2)
        x_2 = x_IN_2 + x_style_2a
        # x_2 = x_IN_2

        x_3 = self.layer3(x_2)
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3a, _, _ = self.SimAM(x_style_3)
        x_3 = x_IN_3 + x_style_3a
        # x_3 = x_IN_3

        x_4 = self.layer4(x_3)
        x_4 = self.avgpool(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        # bottleneck
        end_points['Feature'] = x_4
        x_4 = self.bt(x_4)
        x_4 = self.fc(x_4)

        # no bottleneck
        # end_points['Feature'] = x_4
        # x_4 = self.fc(x_4)

        end_points['Predictions'] = F.softmax(input=x_4, dim=-1)

        if self.training:
            return x_4, end_points, \
                   self.global_avgpool(x_IN_1).view(x_IN_1.size(0), -1), \
                   self.global_avgpool(x_1).view(x_1.size(0), -1), \
                   self.global_avgpool(x_style_1a).view(x_style_1a.size(0), -1), \
                   self.global_avgpool(x_IN_2).view(x_IN_2.size(0), -1), \
                   self.global_avgpool(x_2).view(x_2.size(0), -1), \
                   self.global_avgpool(x_style_2a).view(x_style_2a.size(0), -1), \
                   self.global_avgpool(x_IN_3).view(x_IN_3.size(0), -1), \
                   self.global_avgpool(x_3).view(x_3.size(0), -1), \
                   self.global_avgpool(x_style_3a).view(x_style_3a.size(0), -1), \
                   self.fc3(self.global_avgpool(x_3).view(x_3.size(0), -1))
        else:
            return x_4, end_points, \
                   x_IN_1, x_1, x_style_1a, \
                   x_IN_2, x_2, x_style_2a, \
                   x_IN_3, x_3, x_style_3a
