# standard repos
import os
import re
import time
import math
import json
import random
import itertools

# local modules
from collections import defaultdict

from utils.model import build_model

from utils.loader import load_piecewise
from utils.loader import load_best_config

from utils.train import validate
from utils.train import prepare_csv_data
from utils.train import write_to_csv
from utils.train import save_epoch_params
from utils.train import clear_files
from utils.train import generate_configs

from utils.arguments import train_parse_arguments

# args = train_parse_arguments()
# m = None
# m = re.match(r"cuda:(\d+)", args.device)
# if m is not None:
#     physical_id = m.group(1)
#     os.environ["CUDA_VISIBLE_DEVICES"] = physical_id

from utils.visualize import train_process_visualize
from utils.visualize import plot_batch_of_exp_result

from utils.loader import get_dataloaders

from utils.cmd_printer import print_epoch_status

from utils.hooker import attach_activation_hooks
from utils.hooker import clear_activation_stats
from utils.hooker import save_activation_stats

from utils.MLP_KAN_build import build_MLP_KAN_model

# torch repos
import torch
import torch.nn as nn
import torch.optim as optim

