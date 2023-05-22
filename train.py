# coding=utf-8
import os
import sys
import time
import numpy as np
import argparse
from tqdm import tqdm
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ
from datautil.getdataloader import get_img_dataloader
import torch


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--bottleneck', type=int,
                        default=256, help='bottleneck dim')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--dataset', type=str, default='CSC-NDGS')
    parser.add_argument('--data_dir', type=str, default='/data/sihan.zhu/myfile/dataset/CSC-NDGS/', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--max_epoch', type=int,
                        default=100, help="max iterations")
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="resnet18, resnet18_scsn")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--pm', type=str, default='imp', help='pre-trained model used')
    args = parser.parse_args()
    args.step_per_epoch = 100000
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()
    loss_list = alg_loss_dict(args)
    set_random_seed(args.seed)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc, best_target_acc = 0, 0, 0
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.step_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            step_vals = algorithm.update(minibatches_device)
        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in algorithm.optimizer.param_groups:
                params['lr']=params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========>> epoch %d/%d <<===========' % (epoch, args.max_epoch))
            s = ''
            for item in loss_list:
                s += (item+'_loss:%.4f,' % step_vals[item])
            print(s[:-1])
            s = ''
            for item in acc_type_list:
                acc_record[item] = np.mean(np.array([modelopera.accuracy(
                    algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                s += (item+'_acc:%.4f,' % acc_record[item])
            print(s[:-1])
            if epoch > (args.max_epoch / 5):
                if acc_record['valid'] > best_valid_acc:
                    best_valid_acc = acc_record['valid']
                    target_acc = acc_record['target']
                    path = os.path.join(args.output, 'model.pkl')
                    torch.save(algorithm.state_dict(), path)
                if acc_record['target'] > best_target_acc:
                    best_target_acc = acc_record['target']
            print('total cost time: %.4f' % (time.time()-sss))

    print('best_target acc: %.4f' % (best_target_acc))
    print('DG result: %.4f' % target_acc)

    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('best_target acc: %.4f' % (best_target_acc))
        f.write('target acc:%.4f' % (target_acc))