
# imports
import sys
import gc
from pathlib import Path
from itertools import product

# env variables
DATASET = 'GTSRB'
REPO_PATH = Path().cwd()


from utilities.utils import *
from utilities.pathManager import fetchPaths
from utilities.bdd_parallel_thresholds import run_parallel


# model varient name
POSTFIX = "Adam-32"
lhl_neurons = [50, 80, 200]


flavors = ['raw', 'pca']
subset_neurons = [True, False]
eta = 4

for LOAD_NEURONS, FLAVOR, N in product(subset_neurons, flavors, lhl_neurons):
    print(f'Start Time: {dt.now()}')
    print(DATASET, POSTFIX, N, FLAVOR, LOAD_NEURONS, eta)

    run_parallel((DATASET, POSTFIX, N, FLAVOR, LOAD_NEURONS, eta))
    gc.collect()


