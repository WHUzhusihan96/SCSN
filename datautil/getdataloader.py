# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader


def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                              names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            # obtain all the samples via label
            l = len(tmpdatay)
            if args.split_style == 'strat':
                # better way from shuffle the data in a rate
                lslist = np.arange(l)
                # according to the rate and obtain the train and test index.
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
                # print(len(indextr), len(indexte))
            else:
                # easy way via cutting the data in a rate
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[ted:], indexall[:ted]
            tstep_per_epoch = int(len(indextr)/args.batch_size)
            if tstep_per_epoch < args.step_per_epoch:
                args.step_per_epoch = tstep_per_epoch
                # print("step_per_epoch:  ", args.step_per_epoch)
            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                              names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                              names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte, test_envs=args.test_envs))
    # loaders
    # train_loaders has the source domains (in a rate, eg. 80%)
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]
    # eval_loaders has the all domains:
    # first are the train_loaders' part
    # second are the validation part (the left samples after the train in the source domain)
    # final is the test part (in other words, the target domain)
    # take an example of the CSC-DG and domain 1 is chosen as the target domain
    # so the train loaders has 80% of NO.2 NO.3 NO.4 domains.
    # the eval loaders has 80% of NO.2 NO.3 NO.4 domains and 20% of NO.2 NO.3 NO.4 domains and all of the NO.1 domains
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist+tedatalist]

    return train_loaders, eval_loaders
