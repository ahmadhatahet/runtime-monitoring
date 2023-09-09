from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from pathlib import Path

from utilities.utils import load_json, load_pickle
from utilities.pathManager import fetchPaths
from utilities.pcaFunctions import numComponents
from utilities.MonitorBDD import build_bdd_multi_etas_new


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


def run_parallel(dataset, postfix, lhl, flavor, eta, full, memory):
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

    if full:fn_file_name = lambda x: f"all-thlds-full-{x}-{eta}-{POSTFIX2}.csv"
    else:fn_file_name = lambda x: f"all-thlds-{x}-{eta}-{POSTFIX2}.csv"

    if (path_bdd / fn_file_name('info')).is_file():
        return

    # import lasthidden layer data
    df_train = pd.read_csv(path_lhl_data / f"{filename_postfix}_{flavor}_train.csv")
    df_test = pd.read_csv(path_lhl_data / f"{filename_postfix}_{flavor}_test.csv")

    # load evaluation data or create one
    df_eval = pd.read_csv(path_lhl_data / f"{filename_postfix}_{flavor}_evaluation.csv")

    # select only true classified instances
    if full:
        df_true = pd.concat(
            [
                df_train[df_train["true"] == True].copy(),
                df_test[df_test["true"] == True].copy(),
            ]
        )
    else:
        df_true = df_train[df_train["true"] == True].copy()

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

    # args combinations
    combinations = [
        (thld_name, thld, eta) for (thld_name, thld) in zip(thld_names, thlds)
    ]

    # run parallel build_bdd_multi_etas
    num_cpus = len(combinations) * len(neurons_dict)
    with ProcessPoolExecutor(max_workers=min(num_cpus, 10)) as pool:

        results = [
            pool.submit(build_bdd_multi_etas_new, (
                    df_train.copy(),
                    df_test.copy(),
                    df_true.copy(),
                    df_eval.copy(),
                    dataset,
                    postfix,
                    flavor,
                    neuron_name,
                    selected_neurons,
                    thld_name,
                    thld,
                    eta,
                    full,
                    memory,
                    path_bdd,
                ))
                for thld_name, thld, eta in combinations
                for neuron_name, selected_neurons in neurons_dict.items()
            ]

        # for result in as_completed(results):
        #     subset_name, thld_name, df_bdd_info, df_bdd_scores = result.result()
        #     temp_name = f'{eta}-{subset_name}-{thld_name}-{POSTFIX2}'
        #     df_bdd_info.to_csv(path_bdd / f"all-thlds-full-new-info-{temp_name}.csv", index=False)
        #     df_bdd_scores.to_csv(path_bdd / f"all-thlds-full-new-scores-{temp_name}.csv", index=False)

    # merge all scores and info
    filename_postfix = f"{dataset}_{postfix}"
    # output file name
    POSTFIX2 = flavor.lower()
    POSTFIX2 += f"_{filename_postfix}"
    info_csv_names = []
    scores_csv_names = []
    for thld_name, _, eta in combinations:
        for subset_name, _ in neurons_dict.items():
            temp_name = f'{eta}-{subset_name}-{thld_name}-{POSTFIX2}'
            if full: fn_file_name_thld = lambda x: f"all-thlds-full-{x}-{temp_name}.csv"
            else: fn_file_name_thld = lambda x: f"all-thlds-{x}-{temp_name}.csv"
            info_csv_names.append(fn_file_name_thld('info'))
            scores_csv_names.append(fn_file_name_thld('scores'))

    pd.concat([pd.read_csv(path_bdd / d) for d in info_csv_names]).to_csv(path_bdd / fn_file_name('info'), index=False)
    pd.concat([pd.read_csv(path_bdd / d) for d in scores_csv_names]).to_csv(path_bdd / fn_file_name('scores'), index=False)

    send_email(f'[FINISHED] Train & Test {eta}-{POSTFIX2}')

    print("[" + "=" * 50 + "]")


if __name__ == "__main__":
    ...
