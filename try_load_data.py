import json
import os
from pathlib import Path
import torch
import math
from Bio.PDB.Atom import DisorderedAtom
import pandas as pd
# root_dir = Path('datasets/MultiTask_c03_id09/graphs_new/hetero_fullatom_knn5_spatial4.5_sequential2_train_all')
# data_dir = root_dir / '1a1e.pt'
#
# data_dir2 = root_dir / '1A0A-A.pt'
#
# with open(data_dir, 'rb') as f:
#     data1 = torch.load(f)
# with open(data_dir2, 'rb') as f:
#     data2 = torch.load(f)
#
# print(data1.keys)
# for i in data1.keys:
#     if isinstance(data1[i], torch.Tensor):
#         if data1[i].dtype != data2[i].dtype:
#             print(i, data1[i].dtype, data2[i].dtype)
# def std(data):
#     mean = sum(data) / len(data)
#     var = sum([(x - mean)**2 for x in data]) / len(data)
#     std = math.sqrt(var)
#     return mean, std
# data = [0.2857, 0.2136, 0.2565]
# print(std(data))
result_root = Path('models/hemenet_vallina/hemenet_vallina2023-12-19-21-32-18')
result_dir = result_root / 'res.json'
with open(result_dir, 'r') as f:
    run_results = json.load(f)
results_df = pd.DataFrame(run_results)
results_df.describe().to_csv(result_root / 'summary.csv')