import gc
from pathlib import Path
from itertools import product

DATASET = 'MNIST'
REPO_PATH = Path().cwd()

from utilities.utils import *
from utilities.bdd_parallel_thresholds import run_parallel


# model varient name
POSTFIX = "Adam-256"
lhl_neurons = [30, 60, 100]


flavors = ['raw', 'pca']
subset_neurons = [True, False]
eta = 4

def run():
    for LOAD_NEURONS, FLAVOR, N in product(subset_neurons, flavors, lhl_neurons):
        print(f'Start Time: {dt.now()}')
        print(DATASET, POSTFIX, N, FLAVOR, LOAD_NEURONS, eta)

        run_parallel((DATASET, POSTFIX, N, FLAVOR, LOAD_NEURONS, eta))
        gc.collect()


