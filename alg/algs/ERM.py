# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
import torch.autograd as autograd

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        self.featurizer = get_fea(args)
        self.bottleneck = common_network.feat_bottleneck(
            self.featurizer.in_features, args.bottleneck)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)
        params = [
            {'params': self.featurizer.parameters()},
            {'params': self.bottleneck.parameters()},
            {'params': self.classifier.parameters()}
        ]
        self.optimizer = torch.optim.SGD(params,
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         momentum=args.momentum)

    def update(self, minibatches):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'class': loss.item()}

    def feature(self, x):
        return self.bottleneck(self.featurizer(x))

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))
    # result is the size of  batch_size * num_classes
