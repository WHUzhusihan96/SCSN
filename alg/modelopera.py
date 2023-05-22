# coding=utf-8
import torch
from network import img_network, SCSNnet


def get_fea(args):
    if 'scsn' in args.net:
        net = SCSNnet.ResSCSN_Base(args.net, args.pm)
    elif args.net.startswith('res'):
        net = img_network.ResBase(args.net, args.pm)
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)
            # print(p)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    # print("correct number:{}, total number:{}".format(correct, total))
    return correct / total


def accuracy_infer(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    # print("correct number:{}, total number:{}".format(correct, total))
    return correct / total, correct, total