
import json
import os
import os.path as osp
import random
import time
from argparse import ArgumentParser
from datetime import datetime
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
from torch_geometric.loader import DataLoader
from utils.multitask_data import CustomMultiTaskDataset
from utils.task_models import AffinityModel, MultiTaskModel, PropertyModel

try:
    MODEL_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "models")
except NameError:
    MODEL_DIR = "./models"

if not osp.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# TODO: 1、共享encoder，设置loss函数进行计算
# PPI添加方案1：在训练时，随机打包两个任务的batch加载. 
# PPI添加方案2：在训练时，先训完一个任务的所有batch，再训练另一个任务的batch. Done

class CustomBatchSampler(Sampler[List[int]]):
    def __init__(self, batch_size_main: int = 64, batch_size_aux: int = 4, data_source = None, sample_strategy: int = 0, drop_last: bool = False) -> None:
        """
            sample_strategy: 0 means sampling main task first then sample the aux task.
                             1 means sampling random batches from eicther the main task or the aux task.
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size_main = batch_size_main
        self.batch_size_aux = batch_size_aux
        self.len_main = self.data_source.len_main
        self.drop_last = drop_last
        self.len_aux = self.data_source.len_aux
        self.sample_strategy = sample_strategy
        print("Number of specific data: ", self.data_source.len_main, self.data_source.len_aux)
        print("Number of batches: ", (self.len_main + self.batch_size_main - 1) // self.batch_size_main, (self.len_aux + self.batch_size_aux - 1) // self.batch_size_aux)
    def __iter__(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        rand_train = torch.randperm(self.len_main, generator=generator).tolist()
        rand_aux = (torch.randperm(self.len_aux, generator=generator) + self.len_main).tolist()
        batch = []
        if self.sample_strategy == 0:
            #先随机完主任务，再随机辅助任务
            for item in rand_train:
                batch.append(item)
                if len(batch) == self.batch_size_main:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
            
            batch = []
            for item in rand_aux:
                batch.append(item)
                if len(batch) == self.batch_size_aux:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch        
        elif self.sample_strategy == 1:
            #先随机完辅助任务，再随机主任务
            print("Using Strategy 1")
            for item in rand_aux:
                batch.append(item)
                if len(batch) == self.batch_size_aux:
                    # print(batch)
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                # print(batch)
                yield batch

            batch = []
            for item in rand_train:
                batch.append(item)
                if len(batch) == self.batch_size_main:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
        elif self.sample_strategy == 2:
            #每次随机sample一个batch
            batch_len_train = self.len_main // self.batch_size_main
            batch_len_aux = self.len_aux // self.batch_size_aux
            if not self.drop_last:
                batch_len_train = (self.len_main + self.batch_size_main - 1) // self.batch_size_main
                batch_len_aux = (self.len_aux + self.batch_size_aux - 1) // self.batch_size_aux
            batch_train_current = 0
            batch_aux_current = 0
            for i in range(batch_len_train + batch_len_aux):
                p = random.random()
                if (p < 0.5 or batch_aux_current == batch_len_aux) and batch_train_current != batch_len_train:
                    current_idx_main = batch_train_current * self.batch_size_main
                    if current_idx_main + self.batch_size_main >= self.len_main:
                        batch = rand_train[current_idx_main:]
                    else:
                        batch = rand_train[current_idx_main: current_idx_main + self.batch_size_main]
                    batch_train_current += 1
                    yield batch
                else:
                    current_idx_aux = batch_aux_current * self.batch_size_aux
                    if current_idx_aux + self.batch_size_aux >= self.len_aux:
                        batch = rand_aux[current_idx_aux:]
                    else:
                        batch = rand_aux[current_idx_aux: current_idx_aux + self.batch_size_aux]
                    batch_aux_current += 1
                    yield batch

    def __len__(self):
        if self.drop_last:
            return self.len_main // self.batch_size_main + self.len_aux // self.batch_size_aux
        else:
            return (self.len_main + self.batch_size_main - 1) // self.batch_size_main + (self.len_aux + self.batch_size_aux - 1) // self.batch_size_aux

class LBADataLightning(pl.LightningDataModule):
    def __init__(
        self,
        drop_last,
        batch_size: int = 64,
        num_workers: int = 4,
        train_task: str = 'multi',
        train_split: str = 'train_all',
        val_split: str = 'val',
        test_split: str = 'test',
        hetero = False,
        alpha_only = False,
    ):
        super(LBADataLightning, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.train_task= train_task
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.hetero = hetero
        self.alpha_only = alpha_only
    @property
    def num_features(self) -> int:
        return 9

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        _ = CustomMultiTaskDataset(split='train', task=self.train_task, hetero=self.hetero, alpha_only=self.alpha_only)
        return None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CustomMultiTaskDataset(split=self.train_split, task=self.train_task, hetero=self.hetero, alpha_only=self.alpha_only)
        self.val_dataset = CustomMultiTaskDataset(split=self.val_split, task=self.train_task, hetero=self.hetero, alpha_only=self.alpha_only)
        self.test_dataset = CustomMultiTaskDataset(split=self.test_split, task=self.train_task, hetero=self.hetero, alpha_only=self.alpha_only)

    def train_dataloader(self, shuffle: bool = True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


def choose_monitor(task):
    if task == 'multi':
        monitor = 'val_loss'
        mode = 'min'
    elif task == 'affinity':
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
    print("Config Dict:", args)
    return args


if __name__ == "__main__":

    # Multiprocess Setting to speedup dataloader
    torch.multiprocessing.set_start_method('forkserver')
    torch.set_float32_matmul_precision('high')
    args = get_argparse()
    torch.backends.cudnn.benchmark = False
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
        
    model_dir = osp.join(MODEL_DIR, args.save_dir)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    run_results = []
    seed = args.seed
    for run in range(args.nruns):
        pl.seed_everything(seed, workers=True)
        seed += run
        print(f"Starting run {run} with seed {seed}")
        hetero = True if (args.model_type=='gearnet' or args.model_type=='hemenet') else False,
        train_dataset = CustomMultiTaskDataset(split=args.train_split, task=args.train_task, hetero=hetero, alpha_only=args.alpha_only)
        val_dataset = CustomMultiTaskDataset(split=args.val_split, task=args.train_task, hetero=hetero, alpha_only=args.alpha_only)
        test_dataset = CustomMultiTaskDataset(split=args.test_split, task=args.train_task, hetero=hetero, alpha_only=args.alpha_only)
        trainloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1
        )
        
        valloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1
        )
        
        testloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=1
        )
        
        # datamodule = LBADataLightning(
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers,
        #     drop_last=args.drop_last,
        #     # sample_strategy = args.sample_strategy,
        #     train_task=args.train_task,
        #     train_split=args.train_split,
        #     val_split=args.val_split,
        #     test_split=args.test_split,
        #     hetero=True if (args.model_type=='gearnet' or args.model_type=='hemenet') else False,
        #     alpha_only=args.alpha_only
        #     # auxiliary=None
        # )
        model = model_cls(
            sdim=args.sdim,
            vdim=args.vdim,
            depth=args.depth,
            model_type=args.model_type,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            r_cutoff=4.5,
            num_radial=args.num_radial,
            max_epochs=args.max_epochs,
            factor_scheduler=0.75,
            aggr="mean",
            use_norm=False,
            enhanced=args.enhanced,
            offset_strategy = args.offset_strategy,
            task=args.train_task,
            readout=args.readout
        )
        print(
            f"Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params."
        )
        # 根据不同任务设置选择最优模型的方法
        monitor, mode = choose_monitor(args.train_task)
        
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            filename="model-{epoch:02d}-{val_eloss:.4f}",
            mode=mode,
            save_last=False,
            verbose=True,
        )

        trainer = pl.Trainer(
            devices=device if device != "cpu" else None,
            accelerator="gpu" if device != "cpu" else "cpu",
            max_epochs=args.max_epochs,
            precision=32,
            # amp_backend="native",
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(),
                ModelSummary(max_depth=2)
            ],
            default_root_dir=model_dir,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.batch_accum_grad,
            logger=wandb_logger,
            strategy=DDPStrategy(find_unused_parameters=True),
            num_sanity_val_steps=2
        )

        start_time = datetime.now()

        trainer.fit(model, trainloader, valloader, ckpt_path=args.load_ckpt)

        end_time = datetime.now()
        time_diff = end_time - start_time
        print(f"Training time: {time_diff}")

        # running test set
        _ = trainer.test(ckpt_path="best", dataloaders=testloader)
        res = model.res
        run_results.append(res)

    with open(os.path.join(model_dir, "res.json"), "w") as f:
        json.dump(run_results, f)

    # aggregate over runs..
    results_df = pd.DataFrame(run_results)
    print(results_df.describe())
    # for result in run_results:
    #     wandb_logger.log_metrics(result)