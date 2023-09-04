import os
import time
import yaml
import wandb
import torch
import json
import os.path as osp
import pandas as pd

from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from task_models import MultiTaskModel, PropertyModel, AffinityModel
from torch_geometric.loader import DataLoader

from utils.multitask_data import CustomMultiTaskDataset
# from itertools import cycle
from torch.utils.data import Sampler
from typing import List
from train import LBADataLightning, choose_monitor

try:
    TEST_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "tests")
except NameError:
    TEST_DIR = "./tests"

if not osp.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
    

def get_argparse():
    parser = ArgumentParser(
        description="Main Testing script for Equivariant GNNs on LBA Data."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--hyp_path", type=str, default=None)
    parser.add_argument("--test_name", type=str, default=None)
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
    torch.multiprocessing.set_start_method('forkserver')
    torch.set_float32_matmul_precision('high')
    args = get_argparse()
    device = args.device
    if args.wandb:
        name = args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
        wandb.init(project='gmsl', name=name)
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None
        
    if args.train_task == 'multi':
        model_cls = MultiTaskModel
    elif args.train_task in ['ec', 'go', 'mf', 'bp', 'cc']:
        model_cls = PropertyModel
    elif args.train_task == 'affinity':
        model_cls = AffinityModel
    if args.test_name is not None:
        model_dir = osp.join(TEST_DIR, args.test_name)
    else:
        current_names = os.listdir(TEST_DIR)
        unamed_exps = [i[4:] for i in current_names if 'test' in i]
        if len(unamed_exps) > 0:
            largest = sorted(unamed_exps, key=int)
            current_num = str(int(largest[-1]) + 1)
        else:
            current_num = str(1)
        current_exp = 'test' + current_num
        model_dir = osp.join(TEST_DIR, current_exp)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    datamodule = LBADataLightning(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            train_task=args.train_task,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            gearnet=True if args.model_type=='gearnet' else False,
            alpha_only=args.alpha_only
        )
    

    model = model_cls.load_from_checkpoint(checkpoint_path=args.model_path, 
                                            hyp_path=args.hyp_path,
                                            map_location=None,
                                            offset_strategy=args.offset_strategy
                                           )
    print(
        f"Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params."
        )
    device = args.device
    trainer = pl.Trainer(devices=device if device != "cpu" else None,
            accelerator="gpu" if device != "cpu" else "cpu",)
    start_time = datetime.now()
    trainer.test(model=model, ckpt_path=args.model_path, dataloaders=datamodule, verbose=True)
    end_time = datetime.now()
    time_diff = end_time - start_time
    print(f"Testing time: {time_diff}")
    # Output the testing result
    res = model.res
    with open(os.path.join(model_dir, "res.json"), "w") as f:
        json.dump(res, f)