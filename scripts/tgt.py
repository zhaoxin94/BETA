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
    parser.add_argument("--method", "-m", default="DINE", help="Method")
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
    parser.add_argument("--source", "-s", default='amazon', help="Source")
    parser.add_argument("--target", "-t", default="webcam", help="Target")
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument("--n_trials",
                        "-n",
                        default=1,
                        type=int,
                        help="Repeat times")

    parser.add_argument('--seed', type=int, default=-1)

    # HyperParameters

    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=36)

    parser.add_argument('--no_amp', action='store_false')

    args = parser.parse_args()

    ###############################################################################

    data_root = "/home/zhao/data/DA"
    method_name = args.method.lower()
    if args.dataset == 'office31':
        domains = ["amazon", "dslr", "webcam"]
        data_file = 'office31'
    elif args.dataset == 'officehome':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
        data_file = 'office_home'
    elif args.dataset == 'visda':
        domains = ["synthetic", "real"]
        data_file = 'visda'
        load_epoch = 10
    elif args.dataset == 'cs':
        domains = ["AID", "Merced", "NWPU"]
        data_file = 'cross_scene'
    elif args.dataset == 'domainnet':
        domains = ['painting', 'real', 'sketch']
        data_file = 'domainnet'
        load_epoch = 10
    elif args.dataset == 'minidomainnet':
        domains = ['clipart', 'painting', 'real', 'sketch']
        data_file = 'mini_domainnet'
    else:
        raise ValueError('Unknown Dataset: {}'.format(args.dataset))

    if args.method == 'DINE':
        program = 'train_DINE.py'
    elif args.method == 'BETA':
        program = 'train_BETA.py'
    elif args.method == 'BETA2':
        program = 'train_BETA2.py'
    else:
        raise NotImplementedError

    ###############################################################################

    base_dir = osp.join('output', args.method, args.dataset, args.backbone)

    for i in range(args.n_trials):
        source = args.source
        target = args.target

        output_dir = osp.join(base_dir, source + '_to_' + target, str(i + 1))
        seed = args.seed
        if args.seed < 0:
            seed = seed_hash(args.method, args.backbone, args.dataset, source,
                             target, i)

        src = source[0]
        tgt = target[0]
        load = f'./output/source_only/{args.dataset}/resnet50/{source}/best_val.pth'

        os.system(f'python {program} '
                  f'configs/{data_file}/src_{src}/BETA_{tgt}.py '
                  f'--work-dir {output_dir} '
                  f'--seed {seed} '
                  f'--cfg-options load={load}')
