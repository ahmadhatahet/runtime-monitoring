# import libraries
from pathlib import Path
import sys
from itertools import product
from datetime import datetime as dt

import torchvision.transforms as T

# environment variables
REPO_PATH = Path().cwd()
base = Path(REPO_PATH)
SEED = 42
DATASET = 'FashionMNIST'

# sys.path.append(f"{REPO_PATH}/utilities")
# sys.path.append(f"{REPO_PATH}/{DATASET}/trainingModels")

# # import helper functions
from utils import *
from pcaFunctions import *
from scaleFunctions import *
from pathManager import fetchPaths

# import model
from FashionMNIST_CNN import FashionMNIST_CNN
model = FashionMNIST_CNN

# import train models
from train_models import train_models

# import bdd functions
from bdd_parallel_thresholds import run_parallel

# get essential paths
paths = fetchPaths(base, DATASET)
path = paths[DATASET.lower()]
path_dataset = paths["dataset"]

# # transformers

# tf_train = T.Compose([
#     T.ToTensor(),
#     T.Normalize((0.2850), (0.3200))
# ])


# tf_test = T.Compose([
#     T.ToTensor(),
#     T.Normalize((0.2850), (0.3200))
# ])

# train_data = get_dataset(DATASET, path_dataset, train=True, transform=tf_train)
# test_data = get_dataset(DATASET, path_dataset, train=False, transform=tf_test)


model_setup = {
    "batchnorm": True,
    "dropout": 0.0,
    "first_layer_norm": False,
}
model_config = {
    "optim": "Adam",
    "batch_size": 256,
    "lr": 0.01,
    "epochs": 10,
    "patience": 2,
    "L2": 0.0,
    "L1": 0.0,
}
# model varient variation
models_name = f"{model_config['optim']}-{model_config['batch_size']}"

# neurons
lhl_neurons = [30, 60, 100]

# # train model for each number of neurons
# for n in lhl_neurons:
#     # set last_hidden_neurons for model
#     model_setup["last_hidden_neurons"] = n
#     train_models(model, DATASET, f"{models_name}-{n}", train_data, test_data, model_setup, model_config)

# # run testing bdd thresholds for all models
# # w/o neuron selections with eta = 0
# # flavor ['raw', 'pca']

# flavors = ['raw', 'pca']
# subset_neurons = [True, False]

# max_time = 60 * 15

# for lhl_n, flavor, load_neurons in product(lhl_neurons, flavors, subset_neurons):
#     print(f'Start Time: {dt.now()}')
#     run_parallel((DATASET, models_name, lhl_n, flavor, load_neurons, 0, max_time))


import subprocess
import gc

subprocess.call(['python3', '-V'])

# subprocess.call(
#     ['python3','utilities/bdd_test_thresholds.py',f'-d {DATASET}','-p Adam-256-30','-fl raw','-thld 0.0','-ln 1'])

gc.collect()