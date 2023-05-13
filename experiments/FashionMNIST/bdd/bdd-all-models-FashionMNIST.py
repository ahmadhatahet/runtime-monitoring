
# env variables
REPO_PATH = '/home/ah19/runtime-monitoring'
DATASET = 'FashionMNIST'

import sys
sys.path.append(REPO_PATH)


from utilities.utils import *
from utilities.pathManager import fetchPaths
from utilities.MonitorUnifiedBDD import build_bdd_multi_etas
from utilities.train_models import train_models
from utilities.bdd_threshold import run_parallel


import pandas as pd
import numpy as np
import gc
import copy
from pathlib import Path
from itertools import product



# get essential paths
base = Path().cwd()
paths = fetchPaths(base, DATASET)
path = paths[DATASET.lower()]
path_dataset = paths["dataset"]


# from GTSRB.trainingModels.GTSRB_CNN import GTSRB_CNN
# model =

# tf_train = T.Compose([
#     T.ToTensor(),
#     T.Resize((32, 32)),
#     T.Normalize((0.3359, 0.3110, 0.3224), (0.3359, 0.3110, 0.3224))
# ])


# tf_test = T.Compose([
#     T.ToTensor(),
#     T.Resize((32, 32)),
#     T.Normalize((0.3359, 0.3110, 0.3224), (0.3359, 0.3110, 0.3224))
# ])

# train_data = get_dataset(DATASET, path_dataset, train=True, transform=tf_train)
# test_data = get_dataset(DATASET, path_dataset, train=False, transform=tf_test)


model_setup = {'dropout': 0.2, 'first_layer_norm': False}

model_config = {
    'optim':'Adam',
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 15,
    "patience": 3,
    "L2": 0.0,
    "L1": 0.0
}

# model varient name
models_name = f"{model_config['optim']}-{model_config['batch_size']}"
lhl_neurons = [40, 70, 100]


# In[6]:


# # train model for each number of neurons
# for n in lhl_neurons:
#     # set last_hidden_neurons for model
#     model_setup["last_hidden_neurons"] = n
#     train_models(model, DATASET, f"{models_name}-{n}", train_data, test_data, model_setup, model_config)
#     gc.collect()


# In[7]:


# run testing bdd thresholds for all models
# w/o neuron selections with eta = 0
# flavor ['raw', 'pca']

flavors = ['raw', 'pca']
subset_neurons = [True, False]
max_time = 0
eta = 0
thlds = [0.9, 0, 0.5, 0.7, 0.3]

for thld, load_neurons, flavor, lhl_n in product(thlds, subset_neurons, flavors, lhl_neurons):
    print(f'Start Time: {dt.now()}')
    print(DATASET, models_name, lhl_n, flavor, load_neurons, thld, eta)
    run_parallel(DATASET, models_name, lhl_n, flavor, load_neurons, thld, eta)
    gc.collect()
