import os
import random

import torch
import pickle
from torch_geometric.data.dataset import IndexType
from torch_geometric.data import Dataset
from torch.utils.data import Sampler
import numpy as np
import json
import pytorch_lightning as pl
from typing import Optional, Literal, List
from argparse import Namespace
from torch_geometric.loader import DataLoader
from itertools import cycle

class GMSLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        train_split: str = 'train_all',
        val_split: str = 'val',
        test_split: str = 'test',
        cache_dir: str = None,
        device: Literal['cpu', 'cuda'] = 'cpu',
        seed: int = 817,
        task: str = 'multi'
    ):
        super().__init__()
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.random_state = np.random.RandomState(seed)

    def set_up_ddp(self, local_rank:int, global_rank: int, world_size: int):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        if self.device.type == 'cuda':
            self.device = torch.device(local_rank)

    def setup(self, stage: Optional[str] = None):
        # print(f"Hello from rank {self.trainer.global_rank}")
        self.train_dataset = LiteMultiTaskDataset(graph_cache_dir=self.cache_dir, split=self.train_split, task=self.task)
        if self.task == 'multi':
            self.lba_indices, self.ppi_indices, self.property_indices= self.train_dataset.gen_data_indices(self.train_dataset.complexes)
        self.val_dataset = LiteMultiTaskDataset(graph_cache_dir=self.cache_dir, split=self.val_split, task=self.task)
        self.test_dataset = LiteMultiTaskDataset(graph_cache_dir=self.cache_dir, split=self.test_split, task=self.task)
        self.num_train_batches = (len(self.train_dataset) + self.batch_size - 1) // self.batch_size
        self.set_up_ddp(self.trainer.local_rank, self.trainer.global_rank, self.trainer.world_size)

    def train_dataloader(self):
        # if self.task == 'multi':
        #     print("Using Multitask Dataloader")
        #     return DataLoader(
        #         self.train_dataset,
        #         num_workers=self.num_workers,
        #         pin_memory=True,
        #         persistent_workers=self.num_workers > 0,
        #         batch_sampler=GMSLBatchSampler(lba_data=self.lba_indices, ppi_data=self.ppi_indices,
        #                                  property_data=self.property_indices, batch_size=self.batch_size,
        #                                  num_batches=self.num_train_batches, num_replicas=self.world_size,
        #                                  rank=self.trainer.global_rank, random_state=self.random_state),
        #         prefetch_factor = 2 if self.num_workers > 0 else None,
        #     )
        # else:
        print("Using Normal Dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self, shuffle: bool = False):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def length_check(complx_dir, thres):
    with open(complx_dir, 'rb') as f:
        item = torch.load(f)
    # 两种图，一种multi channel，一种single channel的全原子
    try:
        if item.channel_weights.sum() > thres:
            return False
        else:
            return True
    except:
        if len(item.x) > thres:
            return False
        else:
            return True

class LiteMultiTaskDataset(Dataset):
    """
    The Custom MultiTask Dataset with uniform labels
    """
    graph_cache_dir: str

    def __init__(self, graph_cache_dir= './datasets/MultiTask/processed_graphs/hetero_alpha_only_knn5_spatial4.5_sequential2', split : str = 'train', thres: int = 5000, task: str='multi'):
        super().__init__()
        self.graph_cache_dir = graph_cache_dir + '_' + split
        self.complex_files = os.listdir(self.graph_cache_dir)
        # 可以按需要切数据集
        # if thres is not None:
        #     print(f"Dataset size of split {split} before thres cutting is {len(self.complex_files)}")
        #     self.complex_files = [item for item in self.complex_files if length_check(os.path.join(self.graph_cache_dir, item), thres)]
        #     print(f"Dataset size of split {split} after thres cutting is {len(self.complex_files)}")
        with open('./datasets/MultiTask/uniformed_labels.json', 'r') as f:
            self.label_info = json.load(f)
        # 单任务训练/多任务训练切换
        if task != 'multi' and split not in ['val', 'test']:
            self.filter_task(task, split)
        else:
            print(f"Complexes for task {task} and split {split} is: {len(self.complex_files)}")
    def filter_task(self, task, split):
        task_complexs = []
        short_to_full = {'mf':'molecular_functions', 'bp':'biological_process', 'cc': 'cellular_component'}
        for item in self.complex_files:
            if task=='lba' and self.label_info[item[:-3]]['lba'] != -1:
                task_complexs.append(item)
            elif task=='ppi' and self.label_info[item[:-3]]['ppi'] != -1:
                task_complexs.append(item)
            elif task=='ec' and -1 not in self.label_info[item[:-3]]['ec']:
                task_complexs.append(item)
            elif task in ['mf', 'bp', 'cc'] and -1 not in self.label_info[item[:-3]]['go']:
                valid = True
                for label in self.label_info[item[:-3]]['go']:
                    if len(label[short_to_full[task]]) == 0:
                        valid = False
                if valid:
                    task_complexs.append(item)
        print(f"Complexes for task {task} and {split} is: {len(task_complexs)}")
        self.complex_files = task_complexs
    @property
    def complexes(self):
        return self.complex_files

    def gen_data_indices(self, files):
        lba_indices = []
        ppi_indices = []
        property_indices = []
        for i, item in enumerate(files):
            if self.label_info[item[:-3]]['lba'] != -1:
                lba_indices.append(i)
            elif self.label_info[item[:-3]]['ppi'] != -1:
                ppi_indices.append(i)
            else:
                property_indices.append(i)
        return lba_indices, ppi_indices, property_indices

    def get(self, idx: IndexType):
        with open(os.path.join(self.graph_cache_dir, self.complex_files[idx]), 'rb') as f:  
            item = torch.load(f)
        try:
            _ = item.channel_weights
        except:
            item.channel_weights = torch.zeros(1).long()
        # Tempora fix for single chain affinity labels
        item.affinities = item.affinities.float()
        return item
    
    def len(self) -> int:
        return len(self.complex_files)


class GMSLBatchSampler(Sampler[list[int]]):
    def __init__(self,
                 lba_data: list[int],
                 ppi_data: list[int],
                 property_data: list[int],
                 num_batches: int,
                 num_replicas: int,
                 rank: int,
                 random_state: np.random.RandomState,
                 shuffle: bool = True,
                 batch_size: int | None = None,
                 with_lba = True
                 ):
        if with_lba:
            super().__init__(lba_data + ppi_data + property_data)
        else:
            super().__init__(ppi_data + property_data)
        # print("Hello G", rank, num_replicas, "With lba: ", with_lba)
        ppi_total = len(ppi_data)
        property_total = len(property_data)
        self.with_lba = with_lba
        if with_lba:
            lba_total = len(lba_data)
            self.lba_data = lba_data[rank * (lba_total // num_replicas): min((rank + 1) * (lba_total// num_replicas), lba_total)]
            if shuffle:
                random.shuffle(self.lba_data)
        self.ppi_data = ppi_data[rank * (ppi_total // num_replicas): min((rank + 1) * (ppi_total// num_replicas), ppi_total)]
        self.property_data = property_data[rank * (property_total // num_replicas): min((rank + 1) * (property_total// num_replicas), property_total)]
        if shuffle:
            random.shuffle(self.ppi_data)
            random.shuffle(self.property_data)
        self.num_batches = num_batches
        self.num_replicas = num_replicas
        self.R = random_state # For future random augmentation
        self.rank = rank
        self.batch_size = batch_size

    def __len__(self):
        return self.num_batches // self.num_replicas

    def __iter__(self):
        # For each batch, it must contain at least one sample for each task.
        remain_batches = self.num_batches // self.num_replicas
        batch = []
        next_rank = 0
        if self.with_lba:
            data_to_sample = [self.lba_data, self.ppi_data, self.property_data]# [3 data sources]
            data_idxs = [0, 0, 0]
        else:
            data_to_sample = [self.ppi_data, self.property_data]# [3 data sources]
            data_idxs = [0, 0]
        while remain_batches > 0:
            # lba, ppi, property, 3 types, 2 cut points
            # Each batch contains no more than 1/4 samples that belongs to lba/ppi
            partitions = sorted(random.sample(range(1, max(self.batch_size // 4, 3)), len(data_to_sample) - 1))
            partitions.insert(0, 0)
            partitions.append(self.batch_size)
            num_samples = [partitions[i + 1] - partitions[i] for i in range(0, len(partitions) - 1)]  # [3 types]
            for i in range(len(num_samples)):
                start_idx = data_idxs[i]
                end_idx = data_idxs[i] + num_samples[i]
                if end_idx >= len(data_to_sample[i]) or start_idx >= len(data_to_sample[i]):
                    samples = random.sample(data_to_sample[i], num_samples[i])
                else:
                    samples = data_to_sample[i][data_idxs[i]: data_idxs[i] + num_samples[i]]
                data_idxs[i] += num_samples[i]
                batch.extend(samples)
            assert len(batch) == self.batch_size
            if len(batch) == self.batch_size:
                if next_rank == self.rank:
                    yield batch
                    remain_batches -= 1
                    if remain_batches == 0:
                        break
                next_rank = (next_rank + 1) % self.num_replicas
                batch = []



# 这个sampler是之前做辅助任务时的sampler，暂时没有用到
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

