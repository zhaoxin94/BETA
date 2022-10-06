from email.policy import default
import os
import os.path as osp
import argparse
from collections import defaultdict
import hashlib


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        default="officehome",
        help="Dataset",
        choices=['office31', 'officehome', 'visda', 'domainnet', 'cs'])
    parser.add_argument("--backbone",
                        "-b",
                        default="resnet50",
                        help="Backbone")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument("--n_trials",
                        "-n",
                        default=1,
                        type=int,
                        help="Repeat times")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=-1)

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--vis_tsne', action='store_true')
    parser.add_argument('--vis_score', action='store_true')

    args = parser.parse_args()

    ###############################################################################

    data_root = "/home/zhao/data/bbda"
    method_name = 'source_only'

    if args.dataset == 'office31':
        domains = ["amazon", "dslr", "webcam"]
        data_file = 'office31'
        program = 'train_src_v1.py'
    elif args.dataset == 'officehome':
        domains = ['art', 'clipart', 'product', 'real_world']
        data_file = 'office_home'
        program = 'train_src_v1.py'
    elif args.dataset == 'cs':
        domains = ["AID", "Merced", "NWPU"]
        data_file = 'cross_scene'
        program = 'train_src_v1.py'
    elif args.dataset == 'visda':
        domains = ["synthetic", "real"]
        data_file = 'visda'
        program = 'train_src_v2.py'
    elif args.dataset == 'domainnet':
        domains = ['painting', 'real', 'sketch']
        data_file = 'domainnet'
        program = 'train_src_v1.py'
    elif args.dataset == 'minidomainnet':
        domains = ['clipart', 'painting', 'real', 'sketch']
        data_file = 'mini_domainnet'
        program = 'train_src_v1.py'
    else:
        raise ValueError('Unknown Dataset: {}'.format(args.dataset))

    ###############################################################################

    base_dir = osp.join('output', 'source_only', args.dataset, args.backbone)

    for source in domains:
        for target in domains:
            if target != source:
                break
        if args.dataset == 'visda' and source == 'real':
            print('skip real!')
            continue
        output_dir = osp.join(base_dir, source)
        seed = args.seed
        if args.seed < 0:
            seed = seed_hash(method_name, args.backbone, args.dataset, source)

        src = source[0].upper()
        info_path = f'./data/{data_file}_infos/{src}_list.txt'

        os.system(f'python {program} '
                  f'configs/{data_file}/train_src_base.py '
                  f'--work-dir {output_dir} '
                  f'--seed {seed} '
                  f'--cfg-options src={src} info_path={info_path}')
