import multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path

from utilities.utils import *
from utilities.pathManager import fetchPaths
from utilities.pcaFunctions import applyPCASingle
from utilities.scaleFunctions import applyStandardScalerSingle
from utilities.MonitorUnifiedBDD import build_bdd_multi_etas

from models.mnist_model import MNIST_Model
from models.fashionmnist_model import FashionMNIST_CNN
from models.gtsrb_model import GTSRB_CNN
from models.cifar10_dla import Cifar10_DLA
from models.cifar10_model import Cifar10_CNN

from models.transformers import transformers

models = {
    'mnist': MNIST_Model,
    'fashionmnist': FashionMNIST_CNN,
    'gtsrb': GTSRB_CNN,
#     'cifar10': Cifar10_DLA,
    'cifar10': Cifar10_CNN
}

def run_parallel(args):

    DATASET, TARGET_DATASET, POSTFIX, N, FALVOR, LOAD_NEURONS, eta, memory = args

    POSTFIX = f"{POSTFIX}-{N}"

    # env variables
    FILENAME_POSTFIX = f"{DATASET}_{POSTFIX}"

    # generate paths
    base = Path().cwd()
    paths = fetchPaths(base, DATASET, POSTFIX)
    path_lhl = paths['lhl']
    path_lhl_data = paths['lhl_' + FALVOR]
    path_bdd = paths['bdd'] / FALVOR
    path_bdd.mkdir(exist_ok=True)
    path_target_data = paths['data'].parent / TARGET_DATASET

    configs = load_json(paths['configuration'].parent / 'thresholds.json')
    thresholds = configs['thlds']

    # load config file
    configs = load_json(paths['configuration'])
    config = configs['configuration']

    # model config
    model_setup = configs['model_setup']
    model_config = configs['model_config']

    # saved model
    path_model = next(paths['saved_models'].glob('*.pth.tar'))

    # load trained model
    model_ = models[DATASET.lower()]
    transformer = transformers[DATASET.lower()]

    # device
    CUDA = 0
    GPU_NAME = f'cuda:{CUDA}'
    device = get_device(GPU_NAME)
    torch.cuda.get_device_name(device)

    # load test data
    test_data = get_dataset(TARGET_DATASET, path_target_data, train=False)
    mean_, std_ = transformer['test'].transforms[1].mean, transformer['test'].transforms[1].std
    x_target = torch.stack([test_data[i][0] for i in test_sample]) - mean_ / std_


    # torch 2.0 compile and parallel data training
    model_setup['last_hidden_neurons'] = N
    model = model_(**model_setup).to(device)
    model = torch.compile(model)

    # load model weights
    model.load_state_dict(torch.load(path_model)['model'])

    # eavluation mode
    model.eval()

    # get last hidden layer logits for target data
    logits_target, _ = model.output_last_layer(x_target.to(device))
    logits_target = logits_target.cpu().numpy()
    logits_target = pd.DataFrame(logits_target, columns=[f'x{i}' for i in range(logits_target.shape[1])])

    # project data to pca
    if FALVOR == 'scaler_pca':
        # load sacler object
        sacler_ = load_pickle(path_lhl / f'scaler.pkl')
        # load pca object
        pca_ = load_pickle(path_lhl / f'scaler_pca.pkl')
        # transform data
        logits_target = applyPCASingle(logits_target, sacler_, pca_, N)

    # target data
    df_logits_copy = logits_target.copy()
    df_logits_copy['y'] = 1
    df_logits_copy['true'] = 1


    # output file name
    POSTFIX2 = FALVOR.lower()
    POSTFIX2 += "_neurons" if LOAD_NEURONS else ""
    POSTFIX2 += f"_{FILENAME_POSTFIX}"

    # import lasthidden layer data
    df_train = pd.read_csv(path_lhl_data / f"{FILENAME_POSTFIX}_{FALVOR}_train.csv")
    df_test = pd.read_csv(path_lhl_data / f"{FILENAME_POSTFIX}_{FALVOR}_test.csv")

    # select only true classified instances
    df_true = df_train[df_train["true"] == True].copy()
    df_true = df_true.drop("true", axis=1).reset_index(drop=True)

    # load selected neurons
    neurons_dict = {'None': []}
    if LOAD_NEURONS:
            neurons_dict['gte_mean'] = load_json(path_lhl / "neurons_scaler_pca_gte_mean.json")
            neurons_dict['top_third'] = load_json(path_lhl / "neurons_scaler_pca_top_third.json")
            del neurons_dict['None']

    # define thresholds
    thlds = []
    thld_names = []

    for p in thresholds:
        # round number
        p = np.round(p, 1)
        # generate odd quantiles
        thlds.append(np.quantile(df_true.drop("y", axis=1), p, axis=0))
        thld_names.append(f"qth_{p}")

    # add zero (ReLU)
    thlds.append(np.zeros(df_true.drop("y", axis=1).shape[1]))
    thld_names.append("relu")

    # add mean
    thlds.append(np.mean(df_true.drop("y", axis=1).shape[1], axis=0))
    thld_names.append("mean")


    # loop over two types of selected neurons if LOAD_NEURONS
    for neuron_name, selected_neurons in neurons_dict.items():
        # replace neurons with type of subset
        if LOAD_NEURONS:
            POSTFIX2_TEMP = POSTFIX2.replace('neurons', neuron_name)
        else:
            POSTFIX2_TEMP = POSTFIX2

        # args combinations
        combinations = [(thld_name, thld, eta) for (thld_name, thld) in zip(thld_names, thlds)]

        # drop a combination if the scores already have been collected
        ind_to_pop = []
        for i, (thld_name, *_) in enumerate(combinations):
            if (
                (path_bdd / f"all-thlds-info-{thld_name}-{eta}-{POSTFIX2_TEMP}.csv").is_file()
                and (path_bdd / f"all-thlds-scores-{thld_name}-{eta}-{POSTFIX2_TEMP}.csv").is_file()
            ):
                print(f"[ FOUNDED: {POSTFIX2_TEMP} {thld_name} ]")
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
                [(df_train.copy(), df_test.copy(), df_true.copy(), df_logits_copy.copy(),
                selected_neurons, thld_name, thld, eta, memory, path_bdd)
                for thld_name, thld, eta in combinations])

            pool.close()

            # saving results
            print("[" + "=" * 100 + "]")
            print("> Done All BDDs...")

            df_bdd_info = pd.concat([r[0] for r in results]).reset_index(drop=True)
            df_bdd_info.to_csv(path_bdd / f"all-thlds-info-{eta}-{POSTFIX2_TEMP}.csv", index=False)

            df_bdd_scores = pd.concat([r[1] for r in results]).reset_index(drop=True)
            df_bdd_scores.to_csv(path_bdd / f"all-thlds-scores-{eta}-{POSTFIX2_TEMP}.csv", index=False)

            # replace neurons with type of selected neurons
            for p in path_bdd.glob('*neurons*.csv'):
                p.rename(p.parent / p.name.replace('neurons', neuron_name))

    print("[" + "=" * 100 + "]")
    print("> Finished!")
    print("[" + "*" * 100 + "]")


if __name__ == "__main__":
    ...
