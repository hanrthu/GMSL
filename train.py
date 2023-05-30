import json
import os
import os.path as osp
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Tuple

import math
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary
)
from task_models import MultiTaskModel, PropertyModel, AffinityModel
from torch_geometric.loader import DataLoader

import random
import wandb
from utils.multitask_data import CustomMultiTaskDataset
from utils.data import (
    DATA_DIR,
    CustomLBADataset,
    GNNTransformLBA,
    GNNTransformPPI,
    CustomPPIDataset,
    CustomProteinFunctionDataset,
    CustomAuxDataset,
    PDBBind2016Dataset,
    # LMDBDataset,
)
from atom3d.datasets import LMDBDataset
from itertools import cycle
from torch.utils.data import Sampler
from typing import Sized, List

try:
    PATH = osp.join(osp.dirname(osp.realpath(__file__)), "data")
    MODEL_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "models")
except NameError:
    PATH = "experiments/lba/data"
    MODEL_DIR = "experiments/lba/models"


if not osp.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# TODO: 1、共享encoder，设置loss函数进行计算
# PPI添加方案1：在训练时，随机打包两个任务的batch加载. 
# PPI添加方案2：在训练时，先训完一个任务的所有batch，再训练另一个任务的batch. Done

class CustomBatchSampler(Sampler[List[int]]):
    def __init__(self, batch_size_main: int = 64, batch_size_aux: int = 4, data_source: CustomAuxDataset = None, sample_strategy: int = 0, drop_last: bool = False) -> None:
        """
            sample_strategy: 0 means sampling main task first then sample the aux task.
                             1 means sampling random batches from either the main task or the aux task.
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
        root,
        transform,
        drop_last,
        sample_strategy,
        cache_dir: str = 'ppi_cache.pkl',
        interface_only: bool = False,
        processed: bool = False,
        batch_size: int = 128,
        num_workers: int = 4,
        train_task: str = 'multi',
        train_split: str = 'train_all'
    ):
        super(LBADataLightning, self).__init__()

        self.root = root
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processed = processed
        self.drop_last = drop_last
        self.sample_strategy = sample_strategy
        self.cache_dir = cache_dir
        self.interface = interface_only
        self.train_task= train_task
        self.train_split = train_split
    @property
    def num_features(self) -> int:
        return 9

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self):
        if not self.processed:
            _ = LMDBDataset(osp.join(self.root, "train"), transform=self.transform)
        else:
            _ = CustomLBADataset(split="train")
        return None

    def setup(self, stage: Optional[str] = None):
        # 这里需要改一下任务逻辑，不再是辅助的逻辑了
        if not self.processed:
            # self.lba_train_dataset = LMDBDataset(
            #     osp.join(self.root, "train"), transform=self.transform
            # )
            self.lba_train_dataset = CustomMultiTaskDataset(split=self.train_split, task=self.train_task)
            self.train_dataset = self.lba_train_dataset
            self.val_dataset = CustomMultiTaskDataset(split='val', task=self.train_task)
            self.test_dataset = CustomMultiTaskDataset(split='test', task=self.train_task)
        else:
            # self.train_dataset = CustomLBADataset(split="train")
            self.ppi_train_dataset = CustomPPIDataset(interface=self.interface, ppi_dir=self.ppi_dir, split='train')
            self.lba_train_dataset = CustomLBADataset(split='train')
            self.ec_train_dataset = CustomProteinFunctionDataset(split='train')
            self.train_dataset = CustomAuxDataset(self.lba_train_dataset, self.ppi_train_dataset)
            self.ppi_train_dataset = CustomPPIDataset(interface=self.interface, ppi_dir=self.ppi_dir, split='val')
            self.ppi_train_dataset = CustomPPIDataset(interface=self.interface, ppi_dir=self.ppi_dir, split='test')
            # self.val_dataset = CustomLBADataset(split="val")
            # self.test_dataset = CustomLBADataset(split="test")

    def train_dataloader(self, shuffle: bool = False):
        # if self.auxiliary != None:
        #     batch_sampler = CustomBatchSampler(batch_size_main=self.batch_size, data_source=self.train_dataset, drop_last=self.drop_last, sample_strategy=self.sample_strategy)
        #     # print("Trainset:", self.train_dataset[3506], self.train_dataset[3507])
        #     return DataLoader(
        #         self.train_dataset,
        #         shuffle=shuffle,
        #         num_workers=self.num_workers,
        #         pin_memory=True,
        #         batch_sampler=batch_sampler,
        #         persistent_workers=True
        #     )
        # else:
            # print("Not Using Auxiliary Methods:")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        # if len(self.train_dataset) > len(self.aux_train_dataset):
        #     return zip(cycle(aux_loader), train_loader)
        # else:
        #     return zip(aux_loader, cycle(train_loader))

    def val_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
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
        description="Main training script for Equivariant GNNs on LBA Data."
    )

    # Training Setting
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sdim", type=int, default=100)
    parser.add_argument("--vdim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--model_type", type=str, default="eqgat", choices=["eqgat", "painn", "schnet", "segnn", "egnn", "egnn_edge"]
    )
    parser.add_argument("--cross_ablate", default=False, action="store_true")
    parser.add_argument("--no_feat_attn", default=False, action="store_true")

    parser.add_argument("--nruns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_radial", type=int, default=32)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_accum_grad", type=int, default=1)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--gradient_clip_val", type=float, default=10)
    parser.add_argument("--patience_epochs_lr", type=int, default=10)

    parser.add_argument("--save_dir", type=str, default="base_eqgat")
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--load_ckpt", default=None)
    parser.add_argument("--drop_last", default=False)
    parser.add_argument("--sample_strategy", default=0, type=int)
    parser.add_argument("--cache_dir", default='ppi_cache.pkl', type=str)
    parser.add_argument("--interface_only", default=False, action='store_true')
    parser.add_argument("--enhanced", default=False, action='store_true', help="for EGNN_Edge, choose the enhanced version")
    parser.add_argument("--super_node", default=False, action='store_true', help="Add a supernode or not")
    parser.add_argument("--wandb", default=False, action='store_true', help='Use wandb logger')
    parser.add_argument("--run_name", default='tmp', help='Name a new run')
    parser.add_argument("--train_task", default='multi', help='Choose a task to train the model')
    parser.add_argument("--train_split", default='train_all', help='Choose a split to train the model')
    parser.add_argument("--remove_hydrogen", default=False, action='store_true', help='Choose whether to remove hydrogen elements')
    parser.add_argument("--offset_strategy", default=0, type=int, help='choose the offset strategy for node embedding, 0: no offset, 1:only distinguish ligand and protein, 2: distinguish ligand and every amino acid.')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_argparse()
    device = args.device
    if device != "cpu":
        if ',' not in device:
            device = [int(device)]
        else:
            device = [int(i) for i in device.split(',')]
    if args.wandb:
        wandb.init(project='eqgat', name=args.run_name)
        wandb_logger = WandbLogger()
    else:
        wandb_logger = WandbLogger()
    transform_lba = GNNTransformLBA(
        cutoff=4.5,
        remove_hydrogens=args.remove_hydrogen,
        max_num_neighbors=32,
        supernode=args.super_node,
        offset_strategy=args.offset_strategy
    )

    if args.train_task == 'multi':
        model = MultiTaskModel(
            args=args,
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
            enhanced=args.enhanced,
            offset_strategy = args.offset_strategy,
            task=args.train_task
        )
        print(
            f"Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params."
        )
        model_cls = MultiTaskModel
    elif args.train_task in ['ec', 'go', 'mf', 'bp', 'cc']:
        model = PropertyModel(
            args=args,
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
            enhanced=args.enhanced,
            offset_strategy = args.offset_strategy,
            task=args.train_task
        )
        model_cls = PropertyModel
        print(
            f"Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params."
        )
    elif args.train_task == 'affinity':
        model = AffinityModel(
            args=args,
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
            enhanced=args.enhanced,
            offset_strategy = args.offset_strategy,
            task=args.train_task
        )
        model_cls = AffinityModel
        print(
            f"Model consists of {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params."
        )
        
    model_dir = osp.join(MODEL_DIR, args.save_dir)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    run_results = []
    seed = args.seed
    for run in range(args.nruns):
        pl.seed_everything(seed, workers=True)
        seed += run
        print(f"Starting run {run} with seed {seed}")
        datamodule = LBADataLightning(
            root=DATA_DIR,
            transform=transform_lba,
            processed=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=args.drop_last,
            sample_strategy = args.sample_strategy,
            cache_dir=args.cache_dir,
            interface_only=args.interface_only,
            train_task=args.train_task,
            train_split=args.train_split
            # auxiliary=None
        )
        model = model_cls(
            args=args,
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
            amp_backend="native",
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(),
                ModelSummary(max_depth=2)
            ],
            default_root_dir=model_dir,
            gradient_clip_val=args.gradient_clip_val,
            accumulate_grad_batches=args.batch_accum_grad,
            logger=wandb_logger,
            
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

    with open(os.path.join(model_dir, "res.json"), "w") as f:
        json.dump(run_results, f)

    # aggregate over runs..
    results_df = pd.DataFrame(run_results)
    print(results_df.describe())
    # for result in run_results:
    #     wandb_logger.log_metrics(result)