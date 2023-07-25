# -*- coding: utf-8 -*-
import argparse, os, json

parser = argparse.ArgumentParser()

parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, required=True, help="GPU number")
parser.add_argument('--batch', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
''' loss function '''
parser.add_argument('--loss_fn', type=str, default='ce', 
                    help='Loss function name for training. ex)ce, focal')
parser.add_argument('--loss_weight', type=float, default=12.)
parser.add_argument('--loss_gamma', type=float, default=2.)
parser.add_argument('--loss_alpha', type=float, default=0.25)
parser.add_argument('--label_smoothing', type=float, default=0.1)
''' lr scheduler '''
parser.add_argument('--lr_init', type=float, default=0.001)
parser.add_argument('--lr_min', type=float, default=0.00002)
parser.add_argument('--lr_scheduler', type=str, required=True, 
                    help="learning rate scheduler for training")
parser.add_argument('--cosine_iter', type=int, default=10)
''' experiment infomation '''
parser.add_argument('--exp_name', type=str, default='test', required=True, help='Name of Experiment')
parser.add_argument('--description', type=str, default='test_desc', help='Name of Experiment')
parser.add_argument('--model', type=str, default='densenet', required=True, help='Name of model')
parser.add_argument('--checkpoint', type=str, default=None, help='Pretrained weight checkpoint path')
parser.add_argument('--world_size', default=-1, type=int,help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:40011', type=str,help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,help='distributed backend')
parser.add_argument('--ngpus_per_node', default=1, type=int,help='number of gpu per node')
parser.add_argument('--prefetch_factor', type=int, default=2)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--pin_memory', type=bool, default=True)
parser.add_argument('--data_path', type=str, required=True, help='Data path e.g /data/dk/datasets_CROPS/crops_rescale_0.5_cs_96_64')
parser.add_argument('--save_path', type=str, required=True, help='Data path e.g /data/dk/exp/mass_cls')
parser.add_argument('--optimizer', type=str, required=True, help='optimizer for training')
parser.add_argument('--weight_decay', type=float, required=True, help="weight decay for optimizer. adam: 1e-5")
parser.add_argument('--crop_size', type=str, help="Crop size for input image(space is not allowed) e.g. 72,48,72")
parser.add_argument('--in_channel', type=int, help="Number of channels for input images")
parser.add_argument('--avu', action='store_true', help='Whether use AvUlosss or not')
parser.add_argument('--shift_ratio', type=float, default=0.05, help='The ratio value indicating how much shift in random crop during training.')
parser.add_argument('--sweep', action='store_true', help='Whether sweep or not')

def get_option():
    option = parser.parse_args()
    return option

def save_option(save_dir, option):
    option_path = os.path.join(save_dir, "options.json")
    with open(option_path, 'w') as fp:
        json.dump(option.__dict__, fp, indent=4, sort_keys=True)
