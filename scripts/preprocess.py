from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
import yaml

from utils.multitask_data import CustomMultiTaskDataset

def get_argparse():
    parser = ArgumentParser(
        description="Main training script for Equivariant GNNs on Multitask Data."
    )
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    if args.config != None:
        with open(args.config, 'r') as f:
            content = f.read()
        config_dict = yaml.load(content, Loader=yaml.FullLoader)
        # print("Config Dict:", config_dict)
    else:
        config_dict = {}
    for k, v in config_dict.items():
        setattr(args, k, v)
    print("Config Dict:", args)
    return args

if __name__ == "__main__":
    # Multiprocess Setting to speedup dataloader
    # torch.multiprocessing.set_start_method('forkserver')
    torch.set_float32_matmul_precision('high')
    args = get_argparse()
    device = args.device

    run_results = []
    seed = args.seed
    pl.seed_everything(seed, workers=True)
    hetero = True if (args.model_type == 'gearnet' or args.model_type == 'hemenet') else False,
    train_dataset = CustomMultiTaskDataset(
        split=args.train_split, task=args.train_task, hetero=hetero, alpha_only=args.alpha_only
    )
