
import json
import os
import os.path as osp
import random
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         ModelSummary)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
# from itertools import cycle
from torch.utils.data import Sampler
from utils.datamodule import GMSLDataModule
from torch_geometric.loader import DataLoader
from utils.multitask_data import CustomMultiTaskDataset
from utils.task_models import AffinityModel, MultiTaskModel, PropertyModel
from utils.datamodule import LiteMultiTaskDataset
try:
    MODEL_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "models")
except NameError:
    MODEL_DIR = "./models"

if not osp.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def choose_monitor(task):
    if task == 'multi':
        monitor = 'val_loss'
        mode = 'min'
    elif task in ['lba', 'ppi']:
        monitor = 'val_loss'
        mode = 'min'
    elif task in ['bp', 'mf', 'cc', 'go', 'ec']:
        monitor = 'val_fmax_all'
        mode = 'max'
    return monitor, mode

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
    # print("Config Dict:", args)
    return args

def init_pytorch_settings(args):
    # Multiprocess Setting to speedup dataloader
    torch.multiprocessing.set_start_method('forkserver')
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)

if __name__ == "__main__":
    args = get_argparse()
    init_pytorch_settings(args)
    device = args.device
    name = args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
    if args.wandb:
        wandb.init(project='gmsl_main', name=name)
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None

    if args.model_args['task'] == 'multi':
        model_cls = MultiTaskModel
    elif args.model_args['task'] in ['ec', 'go', 'mf', 'bp', 'cc']:
        model_cls = PropertyModel
    elif args.model_args['task'] in ['lba', 'ppi']:
        model_cls = AffinityModel
        
    model_dir = osp.join(MODEL_DIR, args.save_dir)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    run_results = []
    seed = args.seed
    datamodule = GMSLDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        cache_dir=args.graph_cache_dir,
        device='cuda' if device != 'cpu' else 'cpu',
        seed=args.seed,
        task=args.model_args['task']
    )
    for run in range(args.nruns):
        seed += run
        pl.seed_everything(seed, workers=True)
        print(f"Starting run {run} with seed {seed}")
        model = model_cls(**args.model_args)
        print(
            f"Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params."
        )
        # 根据不同任务设置选择最优模型的方法
        monitor, mode = choose_monitor(args.model_args['task'])

        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            filename="model-{epoch:02d}-{val_loss:.4f}",
            mode=mode,
            save_last=True,
            verbose=True,
            save_top_k=3
        )
        trainer = pl.Trainer(
            devices=device if device != "cpu" else None,
            accelerator="gpu" if device != "cpu" else "cpu",
            max_epochs=args.max_epochs,
            precision='16-mixed' if args.precision!=32 else '32-true',
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(),
                ModelSummary(max_depth=2)
            ],
            default_root_dir=model_dir,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.batch_accum_grad,
            logger=wandb_logger,
            log_every_n_steps=10,
            use_distributed_sampler=False if args.model_args['task']=='multi' else True,
            strategy=DDPStrategy(find_unused_parameters=False),
            num_sanity_val_steps=2,
            benchmark=False,
        )
        start_time = datetime.now()
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.load_ckpt)

        end_time = datetime.now()
        time_diff = end_time - start_time
        print(f"Training time: {time_diff}")

        # running test set
        _ = trainer.test(ckpt_path="best", datamodule=datamodule)
        res = model.res
        run_results.append(res)

    result_dir = Path(model_dir)/ name
    os.makedirs(result_dir, exist_ok=True)
    with open(result_dir / 'res.json', 'w') as f:
        json.dump(run_results, f)

    # aggregate over runs..
    results_df = pd.DataFrame(run_results)
    print(results_df.describe())
    # for result in run_results:
    #     wandb_logger.log_metrics(result)