_base_ = [
    '../finetune_DINE_base.py'
]

# data
src, tgt = 'A', 'C'
info_path = f'./data/office_home_infos/{tgt}_list.txt'
data = dict(
    train=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
    test=dict(
        ds_dict=dict(
            info_path=info_path,
        ),
    ),
)

load = f'./checkpoints/office_home/src_{src}/DINE_{tgt}/last.pth'
