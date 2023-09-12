import os
from pathlib import Path
import torch

root_dir = Path('datasets/MultiTask/graphs_new/hetero_fullatom_knn5_spatial4.5_sequential2_test/')
data_dir = root_dir / '1a1e.pt'

with open(data_dir, 'rb') as f:
    data = torch.load(f)
    print(data)