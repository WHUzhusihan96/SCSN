# coding=utf-8
import os
import sys
import time
import numpy as np
from thop import profile
import argparse
from tqdm import tqdm
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, \
    img_param_init, print_environ
from datautil.getdataloader import get_img_dataloader
import torch
from scipy import io


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
    parser.add_argument('--data_dir', type=str, default='', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="resnet18, resnet18_scsn")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--seed', type=int, default=1218)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--matpath', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.step_per_epoch = 100000
    # args.schuse = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()

    train_loaders, eval_loaders = get_img_dataloader(args)
    # three parts,  3 are the train acc, 3 are the validation acc, and 1 is the test acc.
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()

    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    path = os.path.join(args.output, 'model.pkl')
    print(path)
    checkpoint = torch.load(path)
    algorithm.load_state_dict(checkpoint)

    infer_dict = eval_name_dict['target']
    # infer_dict = eval_name_dict['valid']
    print(infer_dict)
    # print(infer_dict_2)
    loader = [eval_loaders[i] for i in infer_dict]
    correct = 0
    total = 0
    algorithm.eval()
    sss = time.time()
    feature_total = []
    label_total = []
    flag = 0
    for i in infer_dict:
        feature_src = []
        label_src = []
        with torch.no_grad():
            count = 0

            for data in eval_loaders[i]:
                x = data[0].cuda().float()
                y = data[1].cuda().long()
                # print(x.shape)
                p = algorithm.predict(x)
                # b = confusion_matrix(y.cpu(), p.argmax(1).cpu())
                if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float()).sum().item()
                else:
                    correct += (p.argmax(1).eq(y).float()).sum().item()
                total += len(x)

                features = algorithm.feature(x)

                if count == 0:
                    feature_src = features
                    label_src = y
                else:
                    feature_src = torch.cat([feature_src, features], 0)
                    label_src = torch.cat([label_src, y], 0)
                count += 1
        feature_src = feature_src.data.cpu().numpy()
        label_src = label_src.data.cpu().numpy()
        if flag == 0:
            feature_total = feature_src
            label_total = label_src
        else:
            feature_total = np.concatenate((feature_total, feature_src), axis=0)
            label_total = np.concatenate((label_total, label_src), axis=0)
        flag += 1
    print(feature_total.shape)
    print(label_total.shape)
    acc = correct / total
    print('DG result: %.4f' % acc)
    print("correct number: {} / {}".format(int(correct), int(total)))
    print('total cost time: %.4f' % (time.time() - sss))
    mat_name = args.algorithm + '_' + str(args.test_envs) + '_feature_tgt.mat'
    # mat_name = args.algorithm + '_' + str(args.test_envs) + '_feature_src.mat'
    target_path = os.path.join(args.matpath, mat_name)
    print(target_path)
    io.savemat(target_path, {'data': feature_total, 'label': label_total})
