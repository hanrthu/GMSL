# Geometric Multitask Semisupervised Learning for Protein Representation

This is the official repository for Geometric Multitask Semisupervised Learning for Protein Representation(GMSL)

## Overview
This repository contains the scripts of our architecture, and here is the way to run them.

```
git clone git@github.com:hanrthu/GMSL.git
cd GMSL
```

GMSL/ contains the implementation of the Geometric Multitask Semisupervised Learning model with all required submodules. Additionally, we provide implementations of other recent 3D Graph Neural Networks(copied from repository eqgat).

## Installation 
We recommend you to install the dependencies via the fast package management tool [mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html) (you can also replace the command 'mamba' with 'conda' to install them). Generally, GMSL works with Python 3.9.12 and PyTorch version >= 2.0.0
```
mamba env create -f environment.yaml 
mamba activate gmsl
```

After installing the packages, you may encounter the following error in e3nn:
```
TypeError: @e3nn.util.jit.compile_mode can only decorate classes derived from torch.nn.Module
```
You can simply change torch.nn.Module in jit.py into torch.nn.modules.module.Module to solve it. 
## Reproduction
### Dataset 
You can download the pocessed data from the Tsinghua cloud disk:[Multitask.tar.gz](https://cloud.tsinghua.edu.cn/f/bb33cdeaf780472cb8ad/) (~1.5GB) and the label: [uniformed_labels.json](https://cloud.tsinghua.edu.cn/f/57628aaf86044fa7bc38/)(~11MB). Place the downloaded files into the datasets/ folder and extact them.

The number of samples of the original dataset is shown below:

| Dataset | Protein-Ligand | Protein-Protein | EnzymeCommission | GeneOntology |
| :---: | :---: | :---: | :---: | :---: |
| Size | 5208 | 2662 | 18810 | 34944|

The number of processed multi task dataset is shown below (Train_full set contains all the samples with partial labels.):

| Train_full | Val | Test |
| :---: | :---: | :---: |
| 39912 | 531 | 531 |

You can also download the orignal data, exctract it and place them into the datasets/ folder, and process them with preprocess_multi_task_dataet.py to generate the Multitask data yourself. Please check [here](https://github.com/hanrthu/GMSL/blob/master/docs/Process_dataset.md) for documentation of preparing original data to generate multitask dataset.

### Training from scratch
You can use the training script to train a multitask model.

```
python -W ignore train.py --config ./config/gmsl_gearnet_train_all.yaml
```

We provide the hyperparameters for each setting in configuration files. All the configuration files can be found in config/*.yaml.


### Test
You can use the testing script to test the models. For example, to test a trained multitask model, you can provide the checkpoint files and hyperparameter files, and test them as follows:
```
python test.py --config ./config/gmsl_gearnet.yaml --model_path /PATH/TO/YOUR/MODEL --hyp_path /PATH/TO/HYPERPARAMETERS --test_name /TEST/NAME
```
