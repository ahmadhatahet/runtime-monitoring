import multiprocessing as mp
from dd.autoref import BDD

def declare_vars(bdd_):
    """TODO"""
    # generate vars either x0_0 or x0_0 and x0_1 per neuron
    vars_range = neurons if neurons is not None else range(num_neurons)
    v = [f'x{n}' for n in vars_range]

    # add vars to bdd
    [ *map(bdd_.add_var, v) ]

    # generate negative vars
    vars = np.array([ *map(bdd_.var, v) ])
    vars_not = np.array([ ~v for v in vars ])

    return vars, vars_not


def applying_thlds(df):
    """TODO"""
    df_thld = (df >=  thld).astype('int8')
    return df_thld.to_numpy()


def construct_pattern(row, vars, vars_not):
    """TODO"""
    # replace 1 with vars and 0 with vars_not
    expr = np.where( row == 1, vars, vars_not )
    return np.bitwise_and.reduce( expr )


def construct_pattern_parallel(args):
    """TODO"""

    vars_, vars_not_, row = args
    # replace 1 with vars and 0 with vars_not
    expr = np.where( row == 1, vars_, vars_not_ )
    return np.bitwise_and.reduce( expr )



def check_one_pattern(bdd, row, vars, vars_not):
    """TODO"""
    if (roots & construct_pattern(row, vars, vars_not) ) == bdd.false:
        return 0 # means not found
    else:
        return 1 # found it


def score_dataframe(df, bdd_col='bdd'):
    """TODO"""
    df_all_classes = df[['y', 'true']].groupby('y').count().sort_index()
    df_all_classes.columns = ['count']

    df_out_of_pattern_images = df.loc[df[bdd_col] == 0, ['y', bdd_col]].groupby('y').count().sort_index()
    df_out_of_pattern_images.columns = [bdd_col + '_false']

    df_out_of_pattern_misclassified_images = df.loc[(df[bdd_col] == 0) & (df['true'] == False), ['y', bdd_col]].groupby('y').count().sort_index()
    df_out_of_pattern_misclassified_images.columns = [bdd_col + '_false_miss_classified']

    df_scores = df_all_classes.join(df_out_of_pattern_images).join(df_out_of_pattern_misclassified_images)

    del df_out_of_pattern_images, df_out_of_pattern_misclassified_images

    total_images = df_all_classes['count'].sum()
    out_of_pattern_images = (df[bdd_col] == 0).sum()
    out_of_pattern_misclassified_images = ((df['true'] == False) & (df[bdd_col] == 0)).sum()
    df_scores.loc['all', :] = [total_images, out_of_pattern_images, out_of_pattern_misclassified_images]

    # if data frame return 0 rows, a nan will be placed
    df_scores.fillna(0, inplace=True)

    # calculate metrics
    df_scores['outOfPattern'] = df_scores[bdd_col + '_false'] / df_scores['count']
    df_scores['outOfPatternMissClassified'] = df_scores[bdd_col + '_false_miss_classified'] / df_scores[bdd_col + '_false']

    # add mean of all classes
    a1 = df_scores.loc[df_scores.index != 'all', 'outOfPattern'].mean()
    a2 = df_scores.loc[df_scores.index != 'all', 'outOfPatternMissClassified'].mean()
    df_scores.loc['all_mean', :] = [0, 0, total_images, a1, a2]

    # if class is never missclassified and bdd recognize all of his patterns
    # both outOfPattern and outOfPatternMissClassified will be 0
    # so the division will result in NaN
    df_scores['outOfPatternMissClassified'].replace({np.nan:0.0, 0.0:1.0}, inplace=True)
    df_scores['outOfPattern'].replace({np.nan:0.0}, inplace=True)

    # no missclassification for a class
    df_scores[bdd_col + '_false'].replace({np.nan:0.0}, inplace=True)
    df_scores[bdd_col + '_false_miss_classified'].replace({np.nan:0.0}, inplace=True)

    if bdd_col=='bdd':
        return df_scores.reset_index()
    return df_scores


def evaluate_dataframe(self, df, eta=None):
    """TODO"""
    bdd_results = np.zeros(df.shape[0], dtype=np.int8)

    if self.neurons is not None:
        patterns = applying_thlds(df[df.columns[self.neurons]])
    else:
        patterns = applying_thlds(df[df.columns[:self.num_neurons]])

    bdd_results = np.apply_along_axis(check_one_pattern, 1, bdd_bdd, patterns, vars, vars_not)

    # if the function called specifically, return scored df after evaluating
    df['bdd'] = bdd_results

    return score_dataframe(df)



def evaluate_dataframe(self, df, eta=None):
    """TODO"""
    bdd_results = np.zeros(df.shape[0], dtype=np.int8)

    if self.neurons is not None:
        patterns = applying_thlds(df[df.columns[self.neurons]])
    else:
        patterns = applying_thlds(df[df.columns[:self.num_neurons]])

    if self.num_bits == 2:
        patterns = np.apply_along_axis(multi_thlds, 1, patterns)

    bdd_results = np.apply_along_axis(check_one_pattern, 1, patterns)

    if eta is not None:
        df[f'bdd_{eta}'] = bdd_results
        return

    # if the function called specifically, return scored df after evaluating
    df['bdd'] = bdd_results

    return score_dataframe(df)




if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt
    from pympler import asizeof
    import time
    import sys
    from pathlib import Path
    from itertools import product

    # env variables
    REPO_PATH = '/home/ah19/runtime-monitoring'
    DATASET = 'MNIST'
    PREFIX = 'Adam-256-30'
    FILENAME_POSTFIX = f'{DATASET}_{PREFIX}'
    DATA_FALVOR = 'raw'

    sys.path.append(f'{REPO_PATH}/utilities')
    from utils import load_json
    from pathManager import fetchPaths

    # paths
    base = Path(REPO_PATH)
    paths = fetchPaths(base, DATASET)

    path = paths[DATASET.lower()]
    path_bdd = paths['bdd_' + DATA_FALVOR] / FILENAME_POSTFIX

    path_lastHiddenLayer_pca = paths['lastHiddenLayer_pca']
    path_lastHiddenLayer_pca_single = path_lastHiddenLayer_pca / FILENAME_POSTFIX / 'Single'
    # path_lastHiddenLayer_pca_classes = path_lastHiddenLayer_pca / FILENAME_POSTFIX / 'Classes'

    path_lastHiddenLayer = paths['lastHiddenLayer_' + DATA_FALVOR] / FILENAME_POSTFIX


    # import Data
    print('Loading train Data ...')
    df = pd.read_csv(path_lastHiddenLayer / f'{FILENAME_POSTFIX}_train.csv')

    print('Loading test Data ...')
    df_test = pd.read_csv(path_lastHiddenLayer / f'{FILENAME_POSTFIX}_test.csv')


    print('Loading Neurons ...')
    neurons_ = load_json(path_lastHiddenLayer_pca_single / f'{FILENAME_POSTFIX}_neurons.json')


    # split train data
    df_true = df[df['true'] == True].copy()
    df_true = df_true.drop('true', axis=1).reset_index(drop=True)

    # define threshold
    p = 0.95

    thld = np.quantile(df_true.drop('y', axis=1), p, axis=0)
    thld_name = f'qth_{p}'

    # degree of freedom
    eta = 0


    bdd_bdd = BDD()
    bdd_roots = bdd_bdd.false
    n_jobs = 5
    num_neurons = 30
    neurons = None
    if neurons_ is not None:
        neurons = [int(x[1:]) for x in neurons_]
        thld = thld[neurons]

    bdd_vars, bdd_vars_not = declare_vars(bdd_bdd)

    stats = pd.DataFrame({
        'thld': [],
        'df_true': [],
        'build_time': [],
        'size_before_reorder_mb': [],
        'reorder_time': [],
        'size_after_reorder_mb': []
    })


    eval_df_trues=[df_true.copy(), df_test.copy()]

    def add_patterns(args):
        """TODO"""

        (root_, vars_, vars_not_, pattern) = args
        or_expressions = np.apply_along_axis(construct_pattern_parallel, 1, (vars_, vars_not_, pattern))
        root_ |= np.bitwise_or.reduce( or_expressions )

        return root_



    start = time.perf_counter()

    if neurons is not None:
        df_true = df_true[df_true.columns[neurons]].drop_duplicates()
    else:
        df_true = df_true[df_true.columns[:num_neurons]].drop_duplicates()

    df_true_pattern = applying_thlds(df_true)

    # START POOL
    num_procs = mp.cpu_count() if n_jobs == -1 else n_jobs

    # initiate bdds and vars
    print('construct bdds ..')

    bdds = [BDD() for i in range(num_procs)]
    roots = [b.false for b in bdds]
    vars = [declare_vars(b) for b in bdds]
    df_true_patterns = np.array_split(df_true_pattern, num_procs)

    print('start pool ..')
    with mp.Pool(num_procs) as pool:

        results = pool.map(add_patterns, [(root_, vars_, vars_not_, pattern)
                                        for (root_, (vars_, vars_not_), pattern) in zip(roots, vars, df_true_patterns)])

    print('finish pool ..')
    # join bdds
    for r in results:
        bdd_roots |= r
    print('joined bdds ..')

    build_time = round(time.perf_counter() - start, 3)


    row = stats.shape[0]+1
    stats.loc[row, 'df_true'] = 0
    stats.loc[row, 'build_time'] = build_time
    stats.loc[row, 'size_before_reorder_mb'] = round( asizeof.asizeof(bdd_bdd) * 1e-6, 3)


    start = time.perf_counter()
    BDD.reorder(bdd_bdd)
    bdd_reorder_time = round(time.perf_counter() - start, 3)


    stats.loc[row, 'reorder_time'] = bdd_reorder_time
    stats.loc[row, 'size_after_reorder_mb'] = round( asizeof.asizeof(bdd_bdd) * 1e-6, 3)


    # add column for scoring
    if eval_df_trues is not None:
        for eval_df_true in eval_df_trues:
            evaluate_dataframe(eval_df_true, 0)

    stats = stats.loc[stats['df_true'] == eta]