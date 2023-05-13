import pandas as pd
import numpy as np
import sys
from pathlib import Path
from argparse import ArgumentParser

REPO_PATH = Path().cwd()

from utilities.utils import load_json, load_pickle
from utilities.pathManager import fetchPaths
from utilities.pcaFunctions import numComponents
from utilities.MonitorUnifiedBDD import build_bdd_multi_etas


def build_threshold(DATASET, POSTFIX, N, FALVOR, LOAD_NEURONS, THLD, eta, save_path=None):

    # env variables
    FILENAME_POSTFIX = f"{DATASET}_{POSTFIX}"

    # paths
    base = Path(REPO_PATH)
    paths = fetchPaths(base, DATASET, POSTFIX)

    path_bdd = paths['bdd'] / FALVOR
    path_bdd.mkdir(exist_ok=True)
    path_lhl = paths['lhl_' + FALVOR]
    path_lhl_pca = paths['lhl_pca']

    pca_single = None
    if LOAD_NEURONS and FALVOR == 'pca' :
        pca_single = load_pickle(path_lhl_pca / 'pca_single.pkl')
        LOAD_NEURONS = False


    # output file name
    POSTFIX2 = FALVOR.lower()
    POSTFIX2 += "_neurons" if LOAD_NEURONS else ""
    POSTFIX2 += "_components" if pca_single is not None else ""
    POSTFIX2 += f"_{FILENAME_POSTFIX}"


    if THLD == 0: thld_name = f"relu"
    else: thld_name = f"qth_{THLD}"


    # break if info and score file are present
    if not save_path:
        if ((path_bdd / f"info-args-{thld_name}-{POSTFIX2}_CUDD.csv").is_file()
            and (path_bdd / f"scores-args-{thld_name}-{POSTFIX2}_details_CUDD.csv").is_file()):
            print(f"[ FOUNDED: {POSTFIX2} ]")
            return

    # import Data
    df = pd.read_csv(path_lhl / f"{FILENAME_POSTFIX}_train.csv")

    # split train data
    df_true = df[df["true"] == True].copy()
    df_true = df_true.drop("true", axis=1).reset_index(drop=True)

    df_test = pd.read_csv(path_lhl / f"{FILENAME_POSTFIX}_test.csv")

    neurons = None
    if LOAD_NEURONS and pca_single is None:
        neurons = load_json(path_lhl_pca / f"{FILENAME_POSTFIX}_neurons.json")

    if LOAD_NEURONS and pca_single is not None:
        NUM_COMPONENTS = numComponents(pca_single)
        df.drop(df.columns[NUM_COMPONENTS+1:-2], axis=1, inplace=True)
        df_test.drop(df.columns[NUM_COMPONENTS+1:-2], axis=1, inplace=True)
        # true column is dropped, thus only y need to be kept
        df_true.drop(df.columns[NUM_COMPONENTS+1:-1], axis=1, inplace=True)

    # define thresholds
    if THLD == 0: thld = np.zeros(df_true.drop("y", axis=1).shape[1])
    else: thld = np.quantile(df_true.drop("y", axis=1), THLD, axis=0)

    # run build_bdd_multi_etas
    df_bdd_info, df_bdd_scores = build_bdd_multi_etas((
        df, df_test, df_true, neurons,
        thld_name, thld, eta, path_bdd
    ))

    # saving results
    print("[" + "=" * 100 + "]")
    print("> Done ...")

    if not save_path:
        df_bdd_info.to_csv(path_bdd / f"info-args-{thld_name}-{POSTFIX2}_CUDD.csv", index=False)
        df_bdd_scores.to_csv(path_bdd / f"scores-args-{thld_name}-{POSTFIX2}_details_CUDD.csv", index=False)

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
    parser.add_argument('-s', '--save-path', default=0, type=int)

    args = parser.parse_args()

    print(args)

    POSTFIX = '-'.join(args.prefix.split('-')[:-1])
    N = int(args.prefix.split('-')[-1])

    build_threshold(args.dataset, POSTFIX, N, args.flavor, args.load_neurons, args.thld, args.eta, args.save_path)
