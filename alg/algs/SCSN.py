# coding=utf-8
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from alg.modelopera import get_fea
from alg.algs.base import Algorithm
import torch.autograd as autograd
from network import common_network
import math


class SCSN(Algorithm):
    def __init__(self, args):
        super(SCSN, self).__init__(args)
        self.args = args
        self.featurizer = get_fea(args)
        self.optimizer = torch.optim.SGD(self.featurizer.parameters(),
                                         lr=args.lr,
                                         weight_decay=self.args.weight_decay,
                                         momentum=self.args.momentum)

    def update(self, minibatches):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        # forward with the original parameters
        outputs, _, \
        x_IN_1_prob, x_1_prob, x_1_style_prob, \
        x_IN_2_prob, x_2_prob, x_2_style_prob, \
        x_IN_3_prob, x_3_prob, x_3_style_prob, x_3_logits = self.featurizer(all_x)
        cls_loss = 0.01 * F.cross_entropy(x_3_logits, all_y) + F.cross_entropy(outputs, all_y)
        # dis_loss
        dis_criterion = nn.SmoothL1Loss()
        dis_content = dis_criterion(x_IN_1_prob, x_1_prob) \
                      + dis_criterion(x_IN_2_prob, x_2_prob) \
                      + dis_criterion(x_IN_3_prob, x_3_prob)
        dis_style = dis_criterion(x_1_style_prob, x_1_prob) \
                    + dis_criterion(x_2_style_prob, x_2_prob) \
                    + dis_criterion(x_3_style_prob, x_3_prob)
        aux_loss = dis_content - dis_style
        loss = cls_loss + aux_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'class': cls_loss.item(), 'aux_loss': aux_loss.item(), 'loss': loss.item()}

    def feature(self, x):
        x_4, end_points, \
        x_IN_1, x_1, x_1_style, \
        x_IN_2, x_2, x_2_style, \
        x_IN_3, x_3, x_3_style = self.featurizer(x)
        return end_points['Feature']

    def predict(self, x):
        outputs, end_points, \
        x_IN_1, x_1, x_1_style, \
        x_IN_2, x_2, x_2_style, \
        x_IN_3, x_3, x_3_style = self.featurizer(x)
        return outputs