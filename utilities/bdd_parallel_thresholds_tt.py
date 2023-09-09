import multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path

from utilities.utils import load_json, load_pickle
from utilities.pathManager import fetchPaths
from utilities.pcaFunctions import numComponents
from utilities.MonitorBDD import build_bdd_multi_etas


from smtplib import SMTP_SSL

port = 465  # For SSL
password = load_json('/home/ah19/runtime-monitoring/configurations/email-password.json')
password = password['pass']

def send_email(msg):
    sender_email = "cloud243-server@email.com"
    receiver_email = "drahmadhatahet@gmail.com"
    message = f'''
    {msg}'''

    with SMTP_SSL("smtp.gmail.com", port) as server:
        server.login("drahmadhatahet@gmail.com", password)
        server.sendmail(sender_email, receiver_email, message)


def run_parallel(dataset, postfix, lhl, flavor, eta, memory):
    postfix = f"{postfix}-{lhl}"

    # env variables
    filename_postfix = f"{dataset}_{postfix}"

    # generate paths
    base = Path().cwd()
    paths = fetchPaths(base, dataset, postfix)
    path_lhl = paths["lhl"]
    path_lhl_data = paths["lhl_" + flavor]
    path_bdd = paths["bdd"] / flavor
    path_bdd.mkdir(exist_ok=True)

    # load bdd thresholds
    configs_thld = load_json(paths["configuration"].parent / "thresholds.json")
    thresholds = configs_thld["thlds"]

    # output file name
    POSTFIX2 = flavor.lower()
    # POSTFIX2 += "_neurons-full" if is_subset else ""
    POSTFIX2 += f"_{filename_postfix}"

    if (path_bdd / f"all-thlds-info-{eta}-{POSTFIX2}.csv").is_file():
        return

    # import lasthidden layer data
    df_train = pd.read_csv(path_lhl_data / f"{filename_postfix}_{flavor}_train.csv")
    df_test = pd.read_csv(path_lhl_data / f"{filename_postfix}_{flavor}_test.csv")

    # load evaluation data or create one
    df_eval = pd.read_csv(path_lhl_data / f"{filename_postfix}_{flavor}_evaluation.csv")

    # select only true classified instances
    # df_true = df_train[df_train["true"] == True].copy()
    df_true = pd.concat(
        [
            df_train[df_train["true"] == True].copy(),
            df_test[df_test["true"] == True].copy(),
        ]
    )
    df_true = df_true.drop("true", axis=1).reset_index(drop=True)

    # load selected neurons
    neurons_dict = {"None": []}
    if flavor == "raw":
        neurons_dict["gte_mean"] = load_json(
            path_lhl / "neurons_scaler_pca_gte_mean.json"
        )
        neurons_dict["top_third"] = load_json(
            path_lhl / "neurons_scaler_pca_top_third.json"
        )
        # del neurons_dict["None"]

    elif flavor == "scaler_pca":
        pca = load_pickle(path_lhl / "scaler_pca.pkl")
        neurons_dict["components"] = [f"x{i}" for i in range(numComponents(pca))]
        # del neurons_dict["None"]

    # define thresholds
    thlds = []
    thld_names = []

    for p in thresholds:
        # round number
        p = np.round(p, 2)
        # generate odd quantiles
        thlds.append(np.quantile(df_true.drop("y", axis=1), p, axis=0))
        thld_names.append(f"qth_{p}")

    # add zero (ReLU)
    thlds.append(np.zeros(df_true.drop("y", axis=1).shape[1]))
    thld_names.append("relu")

    # add mean
    thlds.append(np.mean(df_true.drop("y", axis=1), axis=0))
    thld_names.append("mean")

    # loop over two types of selected neurons if is_subset

        # replace neurons with type of subset
        # if is_subset:
        #     POSTFIX2_TEMP = POSTFIX2.replace("neurons-full", neuron_name)
        # else:
        #     POSTFIX2_TEMP = POSTFIX2

        # final_csv_filename = lambda type: f"all-thlds-full-{type}-{eta}-{POSTFIX2_TEMP}.csv"

        # if (path_bdd / final_csv_filename("scores")).is_file():
        #     print(f"[ FOUNDED: {dataset} {POSTFIX2_TEMP}]")
        #     continue

    # args combinations
    combinations = [
        (thld_name, thld, eta) for (thld_name, thld) in zip(thld_names, thlds)
    ]


    # run parallel build_bdd_multi_etas
    num_cpus = len(combinations) * len(neurons_dict)
    print("Numper of available CPUs:", min(num_cpus, mp.cpu_count()))
    print("Numper combinations:", num_cpus)
    # print("Numper of available CPUs:", min(num_cpus, 10))
    # print("Numper combinations:", num_cpus)

    pool = mp.Pool(min(num_cpus, 10))

    results = pool.map(
        build_bdd_multi_etas,
        [
            (
                df_train.copy(),
                df_test.copy(),
                df_true.copy(),
                df_eval.copy(),
                neuron_name,
                selected_neurons,
                thld_name,
                thld,
                eta,
                memory,
                path_bdd,
            )
            for thld_name, thld, eta in combinations
            for neuron_name, selected_neurons in neurons_dict.items()
        ],
    )
    pool.close()

    # saving results
    print("[" + "=" * 100 + "]")
    print("> Done All BDDs...")

    df_bdd_info = pd.concat([r[0] for r in results]).reset_index(drop=True)
    df_bdd_info.to_csv(
        path_bdd / f"all-thlds-full-info-{eta}-{POSTFIX2}.csv", index=False
    )

    df_bdd_scores = pd.concat([r[1] for r in results]).reset_index(drop=True)
    df_bdd_scores.to_csv(
        path_bdd / f"all-thlds-full-scores-{eta}-{POSTFIX2}.csv", index=False
    )

    # # replace neurons with type of selected neurons
    # for p in path_bdd.glob("*neurons-full*.csv"):
    #     p.rename(p.parent / p.name.replace("neurons-full", neuron_name))

    send_email(f'[FINISHED] Train & Test {eta}-{POSTFIX2}')

    print("[" + "=" * 100 + "]")
    print("> Finished!")
    print("[" + "*" * 100 + "]")


if __name__ == "__main__":
    ...
