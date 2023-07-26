import gc
from itertools import product

from utilities.utils import *
from utilities.bdd_parallel_thresholds import run_parallel


def run(dataset, postfix, subset_neurons, flavors, lhl_neurons, eta, memory):
    for FLAVOR, LOAD_NEURONS, N in product(flavors, subset_neurons, lhl_neurons):

        # skip pca and selected neurons
        if FLAVOR == 'scaler_pca' and LOAD_NEURONS: continue

        print(f'Start Time: {dt.now()}')
        print(dataset, postfix, N, FLAVOR, LOAD_NEURONS, eta)

        run_parallel((dataset, postfix, N, FLAVOR, LOAD_NEURONS, eta, memory))
        gc.collect()


