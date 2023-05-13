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
    path_bdd = paths['bdd'] / FALVOR
    path_bdd.mkdir(exist_ok=True)
    path_lhl = paths['lhl_' + FALVOR]
    path_lhl_pca = paths['lhl_pca']

    # load pca components
    pca_single = None
    if LOAD_NEURONS and FALVOR == 'pca' :
        pca_single = load_pickle(path_lhl_pca / 'pca_single.pkl')
        LOAD_NEURONS = False

    # output file name
    POSTFIX2 = FALVOR.lower()
    POSTFIX2 += "_neurons" if LOAD_NEURONS else ""
    POSTFIX2 += "_components" if pca_single is not None else ""
    POSTFIX2 += f"_{FILENAME_POSTFIX}"

    # import train data
    df = pd.read_csv(path_lhl / f"{FILENAME_POSTFIX}_train.csv")

    # filter train data to only correctly classfied instances
    df_true = df[df["true"] == True].copy()
    df_true = df_true.drop("true", axis=1).reset_index(drop=True)

    # import test data
    df_test = pd.read_csv(path_lhl / f"{FILENAME_POSTFIX}_test.csv")

    # load selected neurons if raw data
    neurons = []
    if LOAD_NEURONS:
        if pca_single is None:
            neurons = load_json(path_lhl_pca / f"{FILENAME_POSTFIX}_neurons.json")
        # select first n comonents if pca data
        else:
            NUM_COMPONENTS = numComponents(pca_single)
            df.drop(df.columns[NUM_COMPONENTS+1:-2], axis=1, inplace=True)
            df_test.drop(df.columns[NUM_COMPONENTS+1:-2], axis=1, inplace=True)
            # true column is dropped, thus only y need to be kept
            df_true.drop(df.columns[NUM_COMPONENTS+1:-1], axis=1, inplace=True)


    # define thresholds
    thlds = []
    thld_names = []

    # add zero (ReLU) quantile
    thlds.append(np.zeros(df_true.drop("y", axis=1).shape[1]))
    thld_names.append("relu")

    # generate odd quantiles
    thlds.extend([np.quantile(df_true.drop("y", axis=1), np.round(p, 1), axis=0)
         for p in [0.9, 0.7, 0.5, 0.3]])
    thld_names.extend([f"qth_{p}" for p in [0.9, 0.7, 0.5, 0.3]])

    # args combinations
    combinations = [(thld_name, thld, eta) for (thld_name, thld) in zip(thld_names, thlds)]

    # drop a combination if the scores already have been collected
    ind_to_pop = []
    for i, (thld_name, *_) in enumerate(combinations):
        if ((path_bdd / f"info-args-{thld_name}-{POSTFIX2}_CUDD.csv").is_file()
            and (path_bdd / f"scores-args-{thld_name}-{POSTFIX2}_details_CUDD.csv").is_file()):
            print(f"[ FOUNDED: {POSTFIX2} {thld_name} ]")
            ind_to_pop.append(i)

    for i in ind_to_pop[::-1]: combinations.pop(i)

    for c, *_ in combinations: print(c)

    # start the pool
    if len(combinations) > 0:
        # run parallel build_bdd_multi_etas
        print("Numper of available CPUs:", min(len(combinations), mp.cpu_count()))
        print("Numper combinations:", len(combinations))

        pool = mp.Pool(min(len(combinations), mp.cpu_count()))

        results = pool.map(
            build_bdd_multi_etas,
            [(df.copy(), df_test.copy(), df_true.copy(),
              neurons, thld_name, thld, eta, memory, path_bdd)
             for thld_name, thld, eta in combinations])

        pool.close()

        # saving results
        print("[" + "=" * 100 + "]")
        print("> Done All BDDs...")

        df_bdd_info = pd.concat([r[0] for r in results]).reset_index(drop=True)
        df_bdd_info.to_csv(path_bdd / f"all-thlds-info-{POSTFIX2}_CUDD.csv", index=False)

        df_bdd_scores = pd.concat([r[1] for r in results]).reset_index(drop=True)
        df_bdd_scores.to_csv(path_bdd / f"all-thlds-scores-{POSTFIX2}_details_CUDD.csv", index=False)

    print("[" + "=" * 100 + "]")
    print("> Finished!")
    print("[" + "*" * 100 + "]")


if __name__ == "__main__":
    # example run
    DATASET = "GTSRB"
    POSTFIX = "Elastic32"
    N = 40
    FALVOR = 'raw'
    LOAD_NEURONS = True

    eta = 3

    run_parallel((DATASET, POSTFIX, N, FALVOR, LOAD_NEURONS, eta))
