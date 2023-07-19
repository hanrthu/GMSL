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
You may install the dependencies via conda. Generally, GMSL works with Python 3.9.12 and PyTorch version >= 2.0.0
```
conda env create -f environment.yaml 
conda activate gmsl
pip install -e .
```
## Reproduction
### Dataset 
You can download the pocessed data from the Tsinghua Cloud:[Multitask.tar.gz](https://cloud.tsinghua.edu.cn/f/bb33cdeaf780472cb8ad/) (~1.5GB). Place downloaded MultiTask.tar.gz into the datasets/ director and extact the files.

You can also download the orignal data, unzip and place them into the datasets/ folder, and process them with preprocess_multi_task_dataet.py to generate the data yourself.(This will be uploaded later)

### Training from scratch
You can use the training script to 

```
python -W ignore train.py --config ./config/gmsl_gearnet_train_all.yaml
```

We provide the hyperparameters for each setting in configuration files. All the configuration files can be found in config/*.yaml.


### Test
Don't worry, the test script is just on the way~