import multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path

from utilities.utils import load_json, load_pickle
from utilities.pathManager import fetchPaths
from utilities.pcaFunctions import numComponents
from utilities.MonitorUnifiedBDD import build_bdd_multi_etas


def run_parallel(args):

    DATASET, POSTFIX, N, FALVOR, LOAD_NEURONS, eta, memory = args

    POSTFIX = f"{POSTFIX}-{N}"

    # env variables
    FILENAME_POSTFIX = f"{DATASET}_{POSTFIX}"

    # paths
    base = Path().cwd()
    paths = fetchPaths(base, DATASET, POSTFIX)
    path_lhl = paths['lhl']
    path_lhl_data = paths['lhl_' + FALVOR]
    path_bdd = paths['bdd'] / FALVOR
    path_bdd.mkdir(exist_ok=True)

    configs = load_json(paths['configuration'].parent / 'thresholds.json')
    thresholds = configs['thlds']

    # load pca components
    pca_ = None
    if LOAD_NEURONS and FALVOR != 'raw' :
        pca_ = load_pickle(path_lhl / f'{FALVOR}.pkl')


    # output file name
    POSTFIX2 = FALVOR.lower()
    POSTFIX2 += "_neurons" if LOAD_NEURONS else ""
    POSTFIX2 += f"{FALVOR}_components" if pca_ is not None else ""
    POSTFIX2 += f"_{FILENAME_POSTFIX}"

    # import Data
    df_train = pd.read_csv(path_lhl_data / f"{FILENAME_POSTFIX}_{FALVOR}_train.csv")

    # select only true classified
    df_true = df_train[df_train["true"] == True].copy()
    df_true = df_true.drop("true", axis=1).reset_index(drop=True)

    df_test = pd.read_csv(path_lhl_data / f"{FILENAME_POSTFIX}_{FALVOR}_test.csv")

    neurons = []
    if LOAD_NEURONS and pca_ is None:
        neurons = load_json(path_lhl / f"neurons_scaler_pca.json")

    if LOAD_NEURONS and pca_ is not None:
        NUM_COMPONENTS = numComponents(pca_)
        neurons = [f'x{i}' for i in range(NUM_COMPONENTS)]
        # df_train.drop(df_train.columns[NUM_COMPONENTS+1:-2], axis=1, inplace=True)
        # df_test.drop(df_train.columns[NUM_COMPONENTS+1:-2], axis=1, inplace=True)
        # # true column is dropped, thus only y need to be kept
        # df_true.drop(df_train.columns[NUM_COMPONENTS+1:-1], axis=1, inplace=True)


    # define thresholds
    thlds = []
    thld_names = []

    for p in thresholds:
        if p != 0:
            # generate odd quantiles
            thlds.append(np.quantile(df_true.drop("y", axis=1), np.round(p, 1), axis=0))
            thld_names.append(f"qth_{p}")
        else:
            # add zero (ReLU) quantile
            thlds.append(np.zeros(df_true.drop("y", axis=1).shape[1]))
            thld_names.append("relu")


    # args combinations
    combinations = [(thld_name, thld, eta) for (thld_name, thld) in zip(thld_names, thlds)]

    # drop a combination if the scores already have been collected
    ind_to_pop = []
    for i, (thld_name, *_) in enumerate(combinations):
        if (
            (path_bdd / f"info-args-{thld_name}-{eta}-{POSTFIX2}.csv").is_file()
            and (path_bdd / f"scores-args-{thld_name}-{eta}-{POSTFIX2}.csv").is_file()
        ):
            print(f"[ FOUNDED: {POSTFIX2} {thld_name} ]")
            ind_to_pop.append(i)

    for i in ind_to_pop[::-1]: combinations.pop(i)

    # print combination
    for c, *_ in combinations: print(c)

    # start the pool
    if len(combinations) > 0:
        # run parallel build_bdd_multi_etas
        print("Numper of available CPUs:", min(len(combinations), mp.cpu_count()))
        print("Numper combinations:", len(combinations))

        pool = mp.Pool(min(len(combinations), mp.cpu_count()))

        results = pool.map(
            build_bdd_multi_etas,
            [(df_train.copy(), df_test.copy(), df_true.copy(),
              neurons, thld_name, thld, eta, memory, path_bdd)
             for thld_name, thld, eta in combinations])

        pool.close()

        # saving results
        print("[" + "=" * 100 + "]")
        print("> Done All BDDs...")

        df_bdd_info = pd.concat([r[0] for r in results]).reset_index(drop=True)
        df_bdd_info.to_csv(path_bdd / f"all-thlds-info-{eta}-{POSTFIX2}.csv", index=False)

        df_bdd_scores = pd.concat([r[1] for r in results]).reset_index(drop=True)
        df_bdd_scores.to_csv(path_bdd / f"all-thlds-scores-{eta}-{POSTFIX2}.csv", index=False)


    print("[" + "=" * 100 + "]")
    print("> Finished!")
    print("[" + "*" * 100 + "]")


if __name__ == "__main__":
    ...
