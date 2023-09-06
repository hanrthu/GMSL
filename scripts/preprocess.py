from argparse import ArgumentParser
import itertools as it
from pathlib import Path
import pickle

import pytorch_lightning as pl
import torch
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map
import yaml

from utils.multitask_data import GNNTransformMultiTask, chain_uniprot_info

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

def length_check(complx, thres):
    if len(complx['atoms_protein']['element']) > thres:
        return False
    else:
        return True

def correctness_check(chain_uniprot_info, complx):
    #Annotation Correctness Check
    correct = True
    chain_ids = list(set(complx['atoms_protein']['chain']))
    if '-' not in complx['complex_id']:
        for i, id in enumerate(chain_ids):
            if id in chain_uniprot_info[complx['complex_id']]:
                uniprot_id = chain_uniprot_info[complx['complex_id']][id]
                labels_uniprot = complx['labels']['uniprots']
                if uniprot_id not in labels_uniprot:
                    # print("Error, you shouldn't come here!")
                    correct = False
                    # print(complx['complex_id'], labels_uniprot, chain_uniprot_info[complx['complex_id']])
    return correct

def main():
    torch.set_float32_matmul_precision('high')
    args = get_argparse()
    seed = args.seed
    pl.seed_everything(seed, workers=True)
    # cache_dir = os.path.join(self.root_dir, self.cache_dir)
    train_cache_path = Path('datasets/MultiTask/train_all.cache')
    print("Start loading cached Multitask files...")
    with open(train_cache_path, 'rb') as f:
        processed_complexes = pickle.load(f)[:10000]
    print("Complexes Before Checking:", len(processed_complexes))
    print("Checking the dataset...")
    thres = 3000
    processed_complexes = [
        i for i in tqdm(processed_complexes)
        if length_check(i, thres) and correctness_check(chain_uniprot_info, i)
    ]
    print("Dataset size:", len(processed_complexes))
    hetero = True if (args.model_type == 'gearnet' or args.model_type == 'hemenet') else False
    transform_func = GNNTransformMultiTask(hetero=hetero, alpha_only=args.alpha_only)
    print("Transforming complexes...")
    for i in trange(len(processed_complexes), ncols=80):
        processed_complexes[i] = transform_func(processed_complexes[i], 0)
    # process_map(
    #     transform_func,
    #     processed_complexes, it.cycle(range(torch.cuda.device_count())),
    #     max_workers=4, ncols=80, total=len(processed_complexes), chunksize=1,
    # )

if __name__ == "__main__":
    main()
