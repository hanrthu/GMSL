## Geometric Multitask Semisupervised Learning for Protein Representation

This is the official repository for Geometric Multitask Semisupervised Learning for Protein Representation(GMSL)

# EGNN 
CUDA_VISIBLE_DEVICES=3 python train.py --sdim 100 --vdim 16 --depth 3 --nruns 3 --save_dir base_egnn_edge --batch_size 64 --run_name lba_egnn_edge_bs64_withd_ywithe_ep50_double_edge --max_epochs 50 --offset_strategy 1 --super_node --model_type egnn_edge --wandb

CUDA_VISIBLE_DEVICES=0 python train.py --nruns 1 --sdim 100 --vdim 16 --depth 3 --save_dir base_eqgat --batch_size 64 --super_node --wandb --run_name lba_eqgat_pdb16_supernode_bs64_ep50 --offset_strategy 1 --max_epochs 300 --wandb

# Multitask
CUDA_VISIBLE_DEVICES=3 python train.py --sdim 100 --vdim 16 --depth 3 --nruns 3 --save_dir base_egnn_edge --batch_size 32 --run_name multitask_bs32_spnode --max_epochs 50 --offset_strategy 1 --super_node --model_type egnn --train_task multi --train_split train_all --val_split val --test_split test

# EC Only
CUDA_VISIBLE_DEVICES=3 python train.py --sdim 100 --vdim 16 --depth 3 --nruns 3 --save_dir base_egnn_edge --batch_size 32 --run_name multitask_bs32_spnode --max_epochs 50 --offset_strategy 1 --super_node --model_type egnn --train_task ec --train_split train_ec --val_split val_ec --test_split test_ec --device 0


# Gearnet
CUDA_VISIBLE_DEVICES=3 python -W ignore train.py --sdim 100 --vdim 16 --depth 3 --nruns 3 --save_dir gearnet --batch_size 32 --run_name multitask_bs32_gearnet --max_epochs 50 --offset_strategy 1 --model_type gearnet --train_task ec --train_split train_ec --val_split val_ec --test_split test_ec --device 0