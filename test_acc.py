import argparse
import datetime
import json
import random
import time
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from data.OPA import build_dataset
from data.OPA_eval import build_evaluator
from engine import evaluate, train_one_epoch
from models.model import build_ActorCritic
from scorer.model_swin import build_Scorer
from envs_zoo.place_env import Environment

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_args_parser():
    parser = argparse.ArgumentParser('Set params', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=[50], type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    parser.add_argument('--clip_max_norm', default=50, type=float, help='gradient clipping max norm')

    # Scorer
    parser.add_argument('--scorer', default='swin_t', type=str, help="Name of the scorer to use")
    parser.add_argument('--scorer_weight', default='./scorer/model_swin/checkpoint.pth', type=str)

    # ActorCritic
    parser.add_argument('--model', default='model', type=str, help="Name of the model to use")
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--num_actions', default=7, type=int)

    # Environment
    parser.add_argument('--bbox_unit', type=float, default=0.05, help='')
    parser.add_argument('--base_reward', type=float, default=1, help='')
    parser.add_argument('--step_neg_reward_scale', type=float, default=0.01, help='')
    parser.add_argument('--neg_reward', type=float, default=0, help='')
    parser.add_argument('--num_steps', type=int, default=20, help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--max_steps', type=int, default=100, help='number of forward steps in A3C (default: 20)')

    # loss
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=1.00, help='lambda parameter for GAE (default: 1.00)')

    # loss weights
    parser.add_argument('--policy_weight', type=float, default=1, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_weight', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy_weight', type=float, default=0.05, help='value loss coefficient (default: 0.5)')

    # Test
    parser.add_argument('--num_select', default=0, type=int, help='the number of predictions selected for evaluation')
    parser.add_argument('--save_img', default=True, type=bool, help='save test images')

    # Dataset parameters
    parser.add_argument('--dataset_path', default='/media/lingtianxia/Data/Dataset/ObjectPlacement/OPA', type=str)
    parser.add_argument('--train_type', default='train_pos_only', type=str)
    parser.add_argument('--test_type', default='test_pos_single', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--output_dir', default='./result', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='./result/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', default=True, type=bool)
    parser.add_argument('--display_freq', default=10, type=int)
    parser.add_argument('--save_checkpoint_interval', default=50, type=int)

    parser.add_argument('--opa_weight', default='./opa/weight/checkpoint.pth', type=str)

    return parser


def main(args):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build score
    scorer = build_Scorer(args)
    scorer.to(device)
    args.input_dim = scorer.feat_dim

    # build Environment
    environment = Environment(args=args, scorer=scorer, device=device, batch_size=1)

    # build ActorCritic
    model, criterion = build_ActorCritic(args, device=device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_test = build_dataset(mode_type=args.test_type, args=args)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                  collate_fn=utils.collate_fn, drop_last=False, pin_memory=True)
    evaluator = build_evaluator(save_img=args.save_img)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(model, criterion, environment, data_loader_test, evaluator, device, args.output_dir,
                              eval_type=args.test_type, max_steps=args.max_steps, num_select=args.num_select,
                              epoch=args.start_epoch, display_freq=args.display_freq)
        print("test:", test_stats)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir = args.resume[:-len(args.resume.split('/')[-1]) - 1]

    main(args)
