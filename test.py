import os
import json
import time
import yaml
import wandb
import torch
import random
import pandas as pd
import os.path as osp

from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary
)
from task_models import MultiTaskModel, PropertyModel, AffinityModel
from torch_geometric.loader import DataLoader

from utils.multitask_data import CustomMultiTaskDataset
# from itertools import cycle
from torch.utils.data import Sampler
from typing import List