from types import SimpleNamespace
import torch
import numpy as np

from load_data import Data
from main import main

args = {
    'dataset': 'WN18RR',
    'model': 'poincare',
    "num_iterations": 500,
    "nneg": 50,
    "batch_size": 128,
    "lr": 50,
    "dim": 40,
    "cuda": True
}
args = SimpleNamespace(**args)
main(args)