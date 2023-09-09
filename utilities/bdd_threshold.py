import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from utilities.utils import load_json, load_pickle
from utilities.pathManager import fetchPaths
from utilities.pcaFunctions import numComponents
from utilities.MonitorBDD import build_bdd_multi_etas


def build_threshold(DATASET, POSTFIX, FALVOR, LOAD_NEURONS, THLD, eta, memory, save_path=None):

    # env variables
    FILENAME_POSTFIX = f"{DATASET}_{POSTFIX}"


    # paths
    base = Path().cwd()
    paths = fetchPaths(base, DATASET, POSTFIX)

    path_lhl = paths['lhl']
    path_lhl_data = paths['lhl_' + FALVOR]
    path_bdd = paths['bdd'] / FALVOR
    path_bdd.mkdir(exist_ok=True)

    pca_ = None
    if LOAD_NEURONS and FALVOR != 'raw' :
        pca_ = load_pickle(path_lhl / f'{FALVOR}.pkl')
        LOAD_NEURONS = False


    # output file name
    POSTFIX2 = FALVOR.lower()
    POSTFIX2 += "_neurons" if LOAD_NEURONS else ""
    POSTFIX2 += f"{FALVOR}_components" if pca_ is not None else ""
    POSTFIX2 += f"_{FILENAME_POSTFIX}"


    if THLD == 0: thld_name = f"relu"
    else: thld_name = f"qth_{THLD}"


    # break if info and score file are present
    if not save_path:
        if (
            (path_bdd / f"info-args-{thld_name}-{eta}-{POSTFIX2}.csv").is_file()
            and (path_bdd / f"scores-args-{thld_name}-{eta}-{POSTFIX2}.csv").is_file()
        ):
            print(f"[ FOUNDED: {thld_name}-{eta}-{POSTFIX2} ]")
            return

    # import Data
    df_train = pd.read_csv(path_lhl_data / f"{FILENAME_POSTFIX}_{FALVOR}_train.csv")

    # select only true classified
    df_true = df_train[df_train["true"] == True].copy()
    df_true = df_true.drop("true", axis=1).reset_index(drop=True)

    df_test = pd.read_csv(path_lhl_data / f"{FILENAME_POSTFIX}_{FALVOR}_test.csv")

    neurons = None
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
    if THLD == 0: thld = np.zeros(df_true.drop("y", axis=1).shape[1])
    else: thld = np.quantile(df_true.drop("y", axis=1), THLD, axis=0)

    # run build_bdd_multi_etas
    df_bdd_info, df_bdd_scores = build_bdd_multi_etas((
        df_train, df_test, df_true, neurons,
        thld_name, thld, eta, memory, path_bdd
    ))

    # saving results
    print("[" + "=" * 100 + "]")
    print("> Done ...")

    if not save_path:
        df_bdd_info.to_csv(path_bdd / f"info-args-{thld_name}-{eta}-{POSTFIX2}.csv", index=False)
        df_bdd_scores.to_csv(path_bdd / f"scores-args-{thld_name}-{eta}-{POSTFIX2}.csv", index=False)

    print("> Finished!")
    print("[" + "*" * 100 + "]")

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset', required=True, type=str)
    parser.add_argument('-p', '--prefix', required=True, type=str)
    parser.add_argument('-fl', '--flavor', default='raw', type=str)
    parser.add_argument('-thld', default=0, type=float)
    parser.add_argument('-ln', '--load-neurons', default=False, type=int)
    parser.add_argument('-eta', default=0, type=int)
    parser.add_argument('-m', '--memory', default=1_000, type=int)
    parser.add_argument('-s', '--save-path', default=0, type=int)

    args = parser.parse_args()

    print(args)

    POSTFIX = '-'.join(args.prefix.split('-')[:-1])

    build_threshold(args.dataset, POSTFIX, args.flavor, args.load_neurons, args.thld, args.eta, args.memory, args.save_path)
