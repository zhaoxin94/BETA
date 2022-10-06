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
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument("--n_trials",
                        "-n",
                        default=3,
                        type=int,
                        help="Repeat times")

    parser.add_argument('--seed', type=int, default=-1)

    # HyperParameters

    parser.add_argument('--epoch', type=int, default=20)
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
        domains = ['art', 'clipart', 'product', 'real_world']
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
    else:
        raise NotImplementedError

    ###############################################################################
    exp_info = args.exp_name
    if exp_info:
        exp_info = '_' + exp_info

    exp_info += 'epoch={args.epoch}'

    base_dir = osp.join('output', args.method, args.dataset,
                        args.backbone + exp_info)

    for i in range(args.n_trials):
        for source in domains:
            for target in domains:
                if source != target:
                    if args.dataset == 'visda' and source == 'real':
                        print('skip real!')
                        continue
                    output_dir = osp.join(base_dir, source + '_to_' + target,
                                          str(i + 1))
                    seed = args.seed
                    if args.seed < 0:
                        seed = seed_hash(args.method, args.backbone,
                                         args.dataset, source, target, i)

                    # pretrained model path
                    if args.model_path in [
                            'OVA_SO', 'OVA_SOFDA', 'OVA_SOWOWN'
                    ]:
                        model_dir = osp.join('output', args.model_path,
                                             args.dataset + '_' + args.mode,
                                             f'{args.backbone}_lr=0.01',
                                             source, '1')
                        if args.dataset == 'domainnet':
                            model_dir = osp.join(
                                'output', args.model_path,
                                args.dataset + '_' + args.mode,
                                'resnet50_lr=0.01_epochs=10', source, '1')
                    else:
                        raise NotImplementedError

                    src = source[0].upper()
                    info_path = f'./data/{data_file}_infos/{src}_list.txt'

                    os.system(f'python {program} '
                              f'configs/{data_file}/train_src_base.py '
                              f'--work-dir {output_dir} '
                              f'--seed {seed} '
                              f'--cfg-options src={src} info_path={info_path}')
