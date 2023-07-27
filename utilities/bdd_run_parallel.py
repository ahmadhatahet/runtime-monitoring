import gc
from itertools import product

from utilities.utils import *
from utilities.bdd_parallel_thresholds import run_parallel

targets_datasets = {
    'MNIST': 'FashionMNIST',
    'FashionMNIST': 'MNIST',
    'GTSRB': 'Cifar10',
}

def run(dataset, postfix, subset_neurons, flavors, lhl_neurons, eta, memory):
    for N, LOAD_NEURONS, FLAVOR in product(lhl_neurons, subset_neurons, flavors):

        # skip pca and selected neurons
        if FLAVOR == 'scaler_pca' and LOAD_NEURONS: continue

        print(f'Start Time: {dt.now()}')
        print(dataset, postfix, N, FLAVOR, LOAD_NEURONS, eta)

        run_parallel((dataset, targets_datasets[dataset],postfix, N, FLAVOR, LOAD_NEURONS, eta, memory))
        gc.collect()


