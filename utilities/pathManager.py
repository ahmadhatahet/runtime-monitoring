from pathlib import Path

datasets = ["MNIST", "GTSRB", "Cifar10", "FashionMNIST"]

subfolders = {
    "dataset": [],
    "training": ["regularization"],
    "last-hidden-layer": ["raw", "pca"],
    "bdd": [],
    "saved-models": [],
}

def createFolders(base):
    paths = []

    for dataset in datasets:
        for sub1, sub2 in subfolders.items():
            paths.append(base / f'{dataset}/{sub1}')

            for v in sub2:
                if not isinstance(v, dict):
                    paths.append(base / f'{dataset}/{sub1}/{v}')
                else:
                    for k, values in v.items():
                        for v in values:
                            paths.append(base / f'{dataset}/{sub1}/{k}/{v}')

    for p in paths: p.mkdir(parents=True, exist_ok=True)


def fetchPaths(base, dataset, postfix):
    FILENAME_POSTFIX = f'{dataset}_{postfix}'
    paths = {
        'data': base / 'datasets' / dataset,
        'model': base / 'models',
        'saved_models': base / 'experiments' / dataset / 'saved-models' / FILENAME_POSTFIX,
        'bdd': base / 'experiments' / dataset / 'bdd' / FILENAME_POSTFIX,
        'lhl_raw': base / 'experiments' / dataset / 'last-hidden-layer' / 'raw' / FILENAME_POSTFIX,
        'lhl_pca': base / 'experiments' / dataset / 'last-hidden-layer' / 'pca' / FILENAME_POSTFIX / 'single',
    }

    for _, p in paths.items():
        p.mkdir(exist_ok=True, parents=True)

    paths['configuration'] = base / 'configurations' / f'{dataset.lower()}.json'

    return paths

if __name__ == "__main__":
    createFolders( Path('/home/ah19/runtime-monitoring/experiments') )
    # print(fetchPaths( Path('/home/ah19/runtime-monitoring'), 'MNIST' ))

