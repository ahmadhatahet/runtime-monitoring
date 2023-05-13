import multiprocessing as mp
import pandas as pd
import numpy as np
from dd.autoref import BDD

# from pathlib import Path
# from pympler import asizeof
# import time
# import gc
# import pickle
# from datetime import datetime as dt

# number of parallel bdds
NUM_CORES = 8

# get data frame
df_train = pd.read_csv('/home/ah19/runtime-monitoring/MNIST/lastHiddenLayer/raw/MNIST_Adam-256-30/MNIST_Adam-256-30_train.csv')
df_test = pd.read_csv('/home/ah19/runtime-monitoring/MNIST/lastHiddenLayer/raw/MNIST_Adam-256-30/MNIST_Adam-256-30_test.csv')
df_true = df_train.loc[df_train['true'] == True].drop(['y', 'true'], axis=1)

print('Files loaded!')

# convert to pattern
patterns = (df_true > 0).astype('int8').to_numpy()

# seperate to chunks
splitted_patterns = np.array_split(patterns, NUM_CORES, 0)

print('Split Patterns!')

# initiate bdds
bdd = BDD()
root = bdd.false

# add vars to bdd
v = [f'x{n}' for n in range(patterns.shape[1])]
[ *map(bdd.add_var, v) ]

# generate negative vars
vars = np.array([ *map(bdd.var, v) ])
vars_not = np.array([ ~v for v in vars ])

print('BDD!')


# construct pattern
def construct_pattern(row):
    expr = np.where( row == 1, vars, vars_not )
    return np.bitwise_and.reduce( expr )

# main function
def construct_bdd(args):
    print('Constructind BDD!')
    bdd, vars, vars_not, patterns = args

    root_ = bdd.false


    # construct pattern
    def __construct_pattern(row):
        expr = np.where( row == 1, vars, vars_not )
        return np.bitwise_and.reduce( expr )

    # add pattern
    for i in range(patterns.shape[0]):
        root_ |= __construct_pattern(patterns[i])

    return root_

# start pool
pool = mp.Pool(NUM_CORES)

print('Starting Pool!')

# construct multiple bdd
roots = pool.map(construct_bdd, [(bdd, vars, vars_not, sp_pattern) for sp_pattern in splitted_patterns ])

# close pool
pool.close()

print('Done Pool!')

# join bdds
for r in roots:
    root |= r

print('Joining Roots!')

# score patterns
def check_one_pattern(row):
    if (root & construct_pattern(row) ) == bdd.false:
        return 0 # means not found
    else:
        return 1 # found it

test_patterns = (df_test.drop(['y', 'true'], axis=1) > 0).astype('int8').to_numpy()
bdd_results = np.apply_along_axis(check_one_pattern, 1, test_patterns)

df_test['bdd'] = bdd_results

# save scores
def score_dataframe(df, bdd_col='bdd'):
    """TODO"""

    if bdd_col not in df.columns:
        return pd.DataFrame({
                    'y': []
                    ,'count': []
                    ,'_false': []
                    ,'_false_miss_classified': []
                    ,'outOfPattern': []
                    ,'outOfPatternMissClassified': []
                    ,'eta':[]
                })

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

df_score_test = score_dataframe(df_test)

df_score_test.to_csv('/home/ah19/runtime-monitoring/MNIST/bdd/testingThresholds/parallel-bdd.csv', index=False)

print(df_score_test)


print('[===================================================]')
print('Finished!')
print('[===================================================]')