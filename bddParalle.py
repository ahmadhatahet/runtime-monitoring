from argparse import ArgumentParser
from utilities.utils import load_json
from utilities.pathManager import fetchPaths
from pathlib import Path
import gc
from itertools import product

from utilities.utils import load_json
from datetime import datetime as dt
from utilities.bdd_parallel_thresholds import run_parallel


def get_configs(DATASET):
    paths = fetchPaths(Path().cwd(), DATASET, '', False)

    configs = load_json(paths['configuration'])
    config = configs['configuration']

    return config["dataset"], config["lhl_neurons"], config["flavors"], config["subset_neurons"], config["eta"]

if __name__ == "__main__":

    # disable warnings
    import warnings
    warnings.filterwarnings('ignore')

    # python bddParalle.py -d MNIST -p SGD-32 -f 1
    # python bddParalle.py -d FashionMNIST -p AdamW-64 -f 1
    # python bddParalle.py -d GTSRB -p AdamW-32 -f 1
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str)
    parser.add_argument('-p', '--postfix', required=True, type=str)
    parser.add_argument('-f', '--full', required=False, type=int, default=1)
    parser.add_argument('-m', '--memory', required=False, type=int, default=10)

    args = parser.parse_args()

    dataset, lhl_neurons, flavors, subset_neurons, eta = get_configs(args.dataset.lower())

    eta = 0

    for lhl, flavor in product(lhl_neurons, flavors):

        print(f'>> Start Time: {dt.strftime(dt.now(), "%d.%m.%Y - %H:%M")}')
        print('>>', dataset, args.postfix, lhl, flavor, '\tEta:', eta , '\tFull:', args.full)

        run_parallel(dataset, args.postfix, lhl, flavor, eta, args.full, args.memory)
        gc.collect()