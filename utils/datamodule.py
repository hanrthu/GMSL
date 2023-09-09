import os
import random

import torch
from torch_geometric.data.dataset import IndexType
from torch_geometric.data import Dataset
from torch.utils.data import Sampler
import numpy as np
import json
import pytorch_lightning as pl
from typing import Optional, Literal
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
    ):
        super().__init__()

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
        print("hellp")
        self.train_dataset = LiteMultiTaskDataset(graph_cache_dir=self.cache_dir, split=self.train_split)
        self.lba_indices, self.ppi_indices, self.property_indices= self.gen_data_indices(self.train_dataset.complexes)
        self.val_dataset = LiteMultiTaskDataset(graph_cache_dir=self.cache_dir, split=self.val_split)
        self.test_dataset = LiteMultiTaskDataset(graph_cache_dir=self.cache_dir, split=self.test_split)
        self.num_train_batches = (len(self.train_dataset) + self.batch_size - 1) // self.batch_size
        self.set_up_ddp(self.trainer.local_rank, self.trainer.global_rank, self.trainer.world_size)

    def gen_data_indices(self, files):
        with open('./datasets/MultiTask/uniformed_labels.json', 'r') as f:
            label_info = json.load(f)
        lba_indices = []
        ppi_indices = []
        property_indices = []
        for i, item in enumerate(files):
            if label_info[item[:-3]]['lba'] != -1:
                lba_indices.append(i)
            elif label_info[item[:-3]]['ppi'] != -1:
                ppi_indices.append(i)
            else:
                property_indices.append(i)
        return lba_indices, ppi_indices, property_indices

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            batch_sampler=GMSLBatchSampler(lba_data=self.lba_indices, ppi_data=self.ppi_indices,
                                     property_data=self.property_indices, batch_size=self.batch_size,
                                     num_batches=self.num_train_batches, num_replicas=self.world_size,
                                     rank=self.trainer.global_rank, random_state=self.random_state),
            prefetch_factor= 2 if self.num_workers > 0 else None,
        )
        # return DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        #     persistent_workers=self.num_workers > 0,
        # )

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


class LiteMultiTaskDataset(Dataset):
    """
    The Custom MultiTask Dataset with uniform labels
    """
    graph_cache_dir: str

    def __init__(self, graph_cache_dir= './datasets/MultiTask/processed_graphs/hetero_alpha_only_knn5_spatial4.5_sequential2', split : str = 'train'):
        super().__init__()
        self.graph_cache_dir = graph_cache_dir + '_' + split
        self.complex_files = os.listdir(self.graph_cache_dir)

    @property
    def complexes(self):
        return self.complex_files

    def get(self, idx: IndexType):
        with open(os.path.join(self.graph_cache_dir, self.complex_files[idx]), 'rb') as f:  
            item = torch.load(f)
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
                 batch_size: int | None = None,
                 ):
        super().__init__(lba_data + ppi_data + property_data)
        print("Hello G", rank)
        lba_total = len(lba_data)
        ppi_total = len(ppi_data)
        property_total = len(property_data)
        self.lba_data = lba_data[rank * (lba_total // num_replicas): min((rank + 1) * (lba_total// num_replicas), lba_total)]
        self.ppi_data = ppi_data[rank * (ppi_total // num_replicas): min((rank + 1) * (ppi_total// num_replicas), ppi_total)]
        self.property_data = property_data[rank * (property_total // num_replicas): min((rank + 1) * (property_total// num_replicas), property_total)]
        random.shuffle(self.lba_data)
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
        data_to_sample = [self.lba_data, self.ppi_data, self.property_data]# [3 data sources]
        data_idxs = [0, 0, 0]
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