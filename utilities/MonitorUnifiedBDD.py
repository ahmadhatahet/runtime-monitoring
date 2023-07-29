import numpy as np
import pandas as pd
from datetime import datetime as dt
from pathlib import Path
import matplotlib.pyplot as plt
from dd import cudd
import time
import gc

class MonitorBDD:
    def __init__(self, num_neurons, thld_1, thld_2=None, thld_3=None, neurons=[], verbose=False, memory=1_000):

        self.bdd = cudd.BDD(memory_estimate= memory * 2**30)

        # turn off reordering
        self.bdd.configure(reordering=False)

        self.roots = self.bdd.false
        self.num_neurons = num_neurons
        self.verbose = verbose

        self.thld_1 = thld_1
        self.thld_2 = thld_2
        self.thld_3 = thld_3
        self.num_bits = 2 if thld_2 is not None or thld_3 is not None else 1
        self.num_vars = num_neurons * self.num_bits

        self.neurons = np.array(neurons)
        if self.neurons.shape[0] != 0:
            self.neurons = np.array([int(n[1:]) for n in neurons])
            self.thld_1 = self.thld_1[self.neurons]
            self.thld_2 = self.thld_2[self.neurons] if self.thld_2 is not None else None
            self.thld_3 = self.thld_3[self.neurons] if self.thld_3 is not None else None

        self.vars, self.vars_not = self.__declare_vars()

        self.stats = pd.DataFrame({
            'thld': pd.Series(dtype='str'),
            'eta': pd.Series(dtype=np.int8),
            'build_time_min': pd.Series(dtype=np.float16),
            'size_mb': pd.Series(dtype=np.float16),
            'reorder_time_min': pd.Series(dtype=np.float16),
            'num_patterns': pd.Series(dtype=np.int32),
            'num_unique_patterns_%': pd.Series(dtype=np.int32),
            'num_reorder': pd.Series(dtype=np.int8),
            'num_neurons': pd.Series(dtype=np.int16),
            'start_time': pd.Series(dtype='str'),
            'end_time': pd.Series(dtype='str')
        })


    def __declare_vars(self):
        """Add variables to BDD based on nymber of neurons"""

        # generate vars either x0_0 or x0_0 and x0_1 per neuron
        if self.neurons.shape[0] != 0: vars_range = self.neurons
        else: vars_range = range(self.num_neurons)

        vars = [f'x{n}_{i}' for i in range(self.num_bits) for n in vars_range]

        # add vars to bdd
        for v in vars: self.bdd.add_var(v)

        # generate negative vars
        vars = np.array([ *map(self.bdd.var, vars) ])
        vars_not = np.array([ ~v for v in vars ])

        return vars, vars_not


    def __multi_thlds(self, x):
        """TODO"""
        x = x.reshape(x.shape[0], 1)
        x = np.where(x == [0], [0, 0], # 0
             np.where(x == [1], [1, 0], # 1
             np.where(x == [2], [0, 1], # 2
             [1, 1] ) ) )# 3
        return np.reshape(x, x.shape[0] * 2)


    def applying_thlds(self, df):
        """TODO"""

        if self.neurons.shape[0] != 0:
            df = df[df.columns[self.neurons]]
        else:
            df = df[df.columns[:self.num_neurons]]

        df_thld = (df >=  self.thld_1).astype('int8')

        if self.thld_2 is not None:
            df_thld += (df >=  self.thld_2).astype('int8')

        if self.thld_3 is not None:
            df_thld += (df >=  self.thld_3).astype('int8')

        if self.num_bits == 2:
            df_thld = np.apply_along_axis(self.__multi_thlds, 1, df_thld)

        return df_thld.to_numpy()


    def check_pattern_length(self, row):
        """TODO"""
        if self.num_bits == 2:
            assert len(self.vars)/2 == row.shape[0], "ERROR: VARS and ROW do not match!"
        else:
            assert len(self.vars) == row.shape[0], "ERROR: VARS and ROW do not match!"


    def construct_pattern(self, row):
        """TODO"""
        # replace 1 with vars and 0 with vars_not
        expr = np.where( row == 1, self.vars, self.vars_not )
        return np.bitwise_and.reduce( expr )


    def __add_one_pattern(self, row):
        """TODO"""
        self.roots |= self.construct_pattern(row)


    def flip_pattern(self, patterns, eta):
        """flip n-th bit to allow more freedom(false positive)
        if eta = 0 then pattern as is
        if eta = 1 then loop over each bit and force it to one
        eta = 2 loop over 2 bits and flip them ... etc
        drop any duplicate patterns"""
        temp = patterns.copy()

        for nth in range(patterns.shape[1]-eta+1):
            temp[:, nth:nth+eta] = 1

            # compare hamming distance to original
            idx = (patterns ^ temp).sum(1) == eta
            if idx.sum() == 0: continue

            # pick rows with respect to eta
            yield np.unique(temp[idx], axis=0)

            # reset to original value
            temp[:, nth:nth+eta] = patterns[:, nth:nth+eta]


    def add_dataframe(self, df, eta=0, eval_dfs=[]):
        """TODO"""

        row = self.stats.shape[0]+1
        self.stats.loc[row, 'start_time'] = dt.strftime(dt.now(), '%Y-%m-%d %H:%M:%S')

        start = time.perf_counter()
        patterns = self.applying_thlds(df)

        # only unique patterns
        patterns = np.unique(patterns, axis=0)

        self.stats.loc[row, 'num_patterns'] = df.shape[0]
        self.stats.loc[row, 'num_unique_patterns_%'] = round(patterns.shape[0] / df.shape[0], 3) * 100

        for i in range(patterns.shape[0]):
            self.__add_one_pattern(patterns[i])

        build_time = lambda start: round((time.perf_counter() - start) / 60, 3) * 100
        bdd_stats = self.bdd.statistics()
        self.stats.loc[row, 'eta'] = 0
        self.stats.loc[row, 'build_time_min'] = build_time(start)
        self.stats.loc[row, 'size_mb'] = round( bdd_stats['mem'] * 1e-6, 3) # to mb
        self.stats.loc[row, 'reorder_time_min'] = round(bdd_stats['reordering_time'] / 60) * 100 # in minutes
        self.stats.loc[row, 'num_reorder'] = bdd_stats['n_reorderings']
        self.stats.loc[row, 'num_neurons'] = len(self.neurons) if self.neurons.shape[0] != 0 else self.num_neurons
        self.stats.loc[row, 'end_time'] = dt.strftime(dt.now(), '%Y-%m-%d %H:%M:%S')

        # add column for scoring
        if eval_dfs != []:
            for eval_df in eval_dfs:
                self.evaluate_dataframe(eval_df, 0)

        # flip bit
        if eta > 0:
            # evaluate starting from 1 degree of freedom
            for et in range(1, eta+1):

                row = self.stats.shape[0]+1
                self.stats.loc[row, 'start_time'] = dt.strftime(dt.now(), '%Y-%m-%d %H:%M:%S')
                start = time.perf_counter()

                # generate flipped patterns for an eta
                num_flipped_pattern = 0
                num_added_pattern = 0
                for flipped_patterns in self.flip_pattern(patterns, et):
                    num_flipped_pattern += flipped_patterns.shape[0]
                    for pattern in flipped_patterns:
                        if not self.check_one_pattern(pattern):
                            num_added_pattern += 1
                            self.__add_one_pattern(pattern)

                bdd_stats = self.bdd.statistics()
                self.stats.loc[row, 'eta'] = et
                self.stats.loc[row, 'build_time_min'] = build_time(start)
                self.stats.loc[row, 'size_mb'] = round( self.bdd.statistics()['mem'] * 1e-6, 3)
                self.stats.loc[row, 'reorder_time_min'] = round(bdd_stats['reordering_time'] / 60) * 100
                self.stats.loc[row, 'num_reorder'] = bdd_stats['n_reorderings']
                self.stats.loc[row, 'num_neurons'] = len(self.neurons) if self.neurons.shape[0] != 0 else self.num_neurons
                self.stats.loc[row, 'end_time'] = dt.strftime(dt.now(), '%Y-%m-%d %H:%M:%S')
                self.stats.loc[row, 'num_patterns'] = num_flipped_pattern
                self.stats.loc[row, 'num_unique_patterns_%'] = round(num_added_pattern / max(num_flipped_pattern, 1), 3)

                # add column for scoring
                for eval_df in eval_dfs:
                    self.evaluate_dataframe(eval_df, et)

        # return evaluated dataframes
        if eval_dfs != []:
            return eval_dfs

        return


    def check_one_pattern(self, row):
        """convert a pattern to expression and run it against bdd"""
        if (self.roots & self.construct_pattern(row) ) == self.bdd.false:
            return 0 # means not found
        else:
            return 1 # found it


    def evaluate_dataframe(self, df, eta=None):
        """check all rows against the BDD
        return the results as a new column with respect to degree of freedom"""
        bdd_results = np.zeros(df.shape[0], dtype=np.int8)

        patterns = self.applying_thlds(df)

        bdd_results = np.apply_along_axis(self.check_one_pattern, 1, patterns)

        if eta is not None:
            df[f'bdd_{eta}'] = bdd_results
            return

        # if the function called specifically, return scored df after evaluating
        df['bdd'] = bdd_results

        return self.score_dataframe(df)


    def score_dataframe(self, df, bdd_col='bdd'):
        """TODO"""
        # return empty df if false column name was passed
        if bdd_col not in df.columns:
            return pd.DataFrame({
                        'y': []
                        ,'count': []
                        ,'false': []
                        ,'false_misclassified': []
                        ,'false_classified': []
                        ,'outOfPattern': []
                        ,'outOfPatternMisclassified': []
                        ,'outOfPatternClassified': []
                        ,'eta':[]
                    })

        # how many instances per class
        df_all_classes = df[['y', 'true']].groupby('y').count().sort_index()
        df_all_classes.columns = ['count']

        # how many patterns not found
        df_out_of_pattern_images = df.loc[df[bdd_col] == 0, ['y', bdd_col]].groupby('y').count().sort_index()
        df_out_of_pattern_images.columns = [bdd_col + '_false']

        # how many patterns not found and misclassified
        df_out_of_pattern_misclassified_images = df.loc[(df[bdd_col] == 0) & (df['true'] == False), ['y', bdd_col]].groupby('y').count().sort_index()
        df_out_of_pattern_misclassified_images.columns = [bdd_col + '_false_misclassified']

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

        # more misclassfied and undetected pattern means the monitor is correctly detecting unfamiliar patterns
        df_scores['outOfPatternMisclassified'] = df_scores[bdd_col + '_false_misclassified'] / df_scores[bdd_col + '_false']


        # add mean of all classes
        a1 = df_scores.loc[df_scores.index != 'all', 'outOfPattern'].mean()
        a2 = df_scores.loc[df_scores.index != 'all', 'outOfPatternMisclassified'].mean()

        # if class is never Misclassified and bdd recognize all of its patterns
        # both outOfPattern and outOfPatternMisclassified will be 0
        # so the division will result in NaN, thus will be replaced by zero
        # because we don't know how the monitor will react once the class's data start to get outdated
        df_scores['outOfPatternMisclassified'].replace({np.nan:0.0, 0.0:1.0}, inplace=True)
        df_scores['outOfPattern'].replace({np.nan:0.0}, inplace=True)

        # no missclassification for a class
        df_scores[bdd_col + '_false'].replace({np.nan:0.0}, inplace=True)
        df_scores[bdd_col + '_false_misclassified'].replace({np.nan:0.0}, inplace=True)

        # how many pattern were undetected but correctly classified
        df_scores[bdd_col + '_false_classified'] = df_scores[bdd_col + '_false'] - df_scores[bdd_col + '_false_misclassified']
        df_scores['outOfPatternClassified'] = 1 - df_scores['outOfPatternMisclassified']

        out_of_pattern_classified_images = out_of_pattern_images - out_of_pattern_misclassified_images
        df_scores.loc['all', bdd_col + '_false_classified'] = out_of_pattern_classified_images
        # df_scores['outOfPatternClassified'].replace({np.nan:0.0, 0.0:1.0}, inplace=True)
        a3 = df_scores.loc[df_scores.index != 'all', 'outOfPatternClassified'].mean()
        df_scores.loc['all_mean', :] = [total_images, out_of_pattern_images, out_of_pattern_misclassified_images, 0, a1, a2, a3]

        # reorder columns
        df_scores = df_scores[['count', bdd_col + '_false', bdd_col + '_false_misclassified', bdd_col + '_false_classified',
                            'outOfPattern','outOfPatternMisclassified','outOfPatternClassified']]

        if bdd_col=='bdd':
            return df_scores.reset_index()
        return df_scores


    def score_dataframe_multi_eta(self, df, eta):
        """TODO"""
        df_scores = pd.DataFrame()

        for et in range(eta+1):
            temp = self.score_dataframe(df, f'bdd_{et}')
            temp['eta'] = et
            temp.columns = [*map(lambda x: x.replace(f'bdd_{et}_', ''), temp.columns)]
            df_scores = pd.concat([df_scores, temp])
            del temp

        return df_scores.reset_index()


    def plot_stats(self, df, stage, true=True, save_folder=None, prefix=None):
        """TODO"""
        df = df.loc[df['true'] == true].set_index('y')
        mean_ = df['outOfPattern'].mean().round(3)

        neurons = f' Number of neuron: {len(self.neurons)}' if self.neurons.shape[0] != 0 else None
        title = f'{stage} - {true} - #out of pattern: {mean_}{neurons}\n'
        filename = f'{stage.lower()}_{true}_outOfPattern'
        color = 'teal' if true else 'orange'

        df['outOfPattern'].plot(
            kind='bar', title=title, legend=False, color=color, xlabel='', hatch='x', edgecolor='black')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.axhline(mean_, color='red', linewidth=2, linestyle='--')


        if save_folder is not None:
            plt.savefig(save_folder / f'bdd_scores_{filename}_{prefix}.jpg', dpi=150, transparent=False)

        plt.show()


def build_bdd_multi_etas(args):
    df_train, df_test, df_true, df_logits, neurons, thld_name, thld, eta, memory, save_path = args

    from dd.autoref import BDD

    # construcr MonitorBDD
    patterns = MonitorBDD( df_true.shape[1]-1, thld, neurons=neurons, memory=memory)
    print(f'Threshold: {thld_name}, eta: {eta}')

    # build
    patterns.add_dataframe( df_true, eta, eval_dfs=[df_train, df_test, df_logits] )

    # collect scores
    df_bdd_info = patterns.stats.copy()
    df_bdd_info['thld'] = thld_name

    df_train_scores = patterns.score_dataframe_multi_eta(df_train, eta)
    df_test_scores = patterns.score_dataframe_multi_eta(df_test, eta)
    df_logit_scores = patterns.score_dataframe_multi_eta(df_logits, eta)
    df_train_scores['stage'] = 'train'
    df_test_scores['stage'] = 'test'
    df_logit_scores['stage'] = 'evaluation'

    # combine scores
    df_bdd_scores = pd.concat([df_train_scores, df_test_scores, df_logit_scores]).reset_index(drop=True)
    df_bdd_scores['thld'] = thld_name



    # save data and scores
    if save_path is not None:
        temp_name = thld_name

        if len(neurons) > 0 and save_path.name == 'raw':
            temp_name += "-neurons"
        # this combination is not valid
        # if len(neurons) > 0 and save_path.name == 'scaler_pca':
        #     temp_name += "-neurons-pca"

        # some bdds are too big to save
        # thus the processed data and bdd results will be saved instead
        # temp_path = Path('/tmp/ah19') / save_path.parent.name / save_path.name
        # temp_path.mkdir(parents=True, exist_ok=True)

        # multiprocessing can not save serialized objects
        # from utilities.utils import save_pickle
        # save_pickle(temp_path / f'{temp_name}-{eta}.pkl', patterns)

        # save scores
        # df_bdd_info.to_csv(save_path / f'info-{thld_name}-{eta}-{temp_name}.csv', index=False)
        # df_bdd_scores.to_csv(save_path / f'scores-{thld_name}-{eta}-{temp_name}.csv', index=False)

        # apply threshold
        if patterns.neurons.shape[0] != 0:
            idx_col_keep = patterns.neurons
            idx_col_delete = np.delete(df_train.columns[:patterns.num_neurons], idx_col_keep)
        else:
            idx_col_keep = np.arange(patterns.num_neurons)
            idx_col_delete = np.delete(df_train.columns[:patterns.num_neurons], idx_col_keep)


        df_train[df_train.columns[idx_col_keep]] = patterns.applying_thlds(df_train)
        df_train.drop(idx_col_delete, axis=1, inplace=True)

        df_test[df_test.columns[idx_col_keep]] = patterns.applying_thlds(df_test)
        df_test.drop(idx_col_delete, axis=1, inplace=True)

        df_logits[df_logits.columns[idx_col_keep]] = patterns.applying_thlds(df_logits)
        df_logits.drop(idx_col_delete, axis=1, inplace=True)

        # save processed data and the BDD result
        df_train.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_train.csv', index=False)
        df_test.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_test.csv', index=False)
        df_logits.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_evaluation.csv', index=False)

    # delete variables
    del BDD, patterns
    del df_train_scores, df_test_scores
    gc.collect()

    print(f'> Done! [ {thld_name} - eta: {eta} ]')

    return df_bdd_info, df_bdd_scores



def build_bdd(args):
    df_train, df_test, df_true, neurons, thld_name, thld, eta, save_path = args

    from dd.autoref import BDD

    # construcr MonitorBDD
    patterns = MonitorBDD( df_true.shape[1]-1, thld, neurons=neurons )
    print(f'{thld_name} - eta: {eta}')

    # build
    patterns.add_dataframe( df_true, eta)

    # collect scores
    df_bdd_info = patterns.stats.copy()
    df_bdd_info['thld'] = thld_name

    df_train_scores = patterns.evaluate_dataframe(df_train)
    df_test_scores = patterns.evaluate_dataframe(df_test)

    df_train_scores['stage'] = 'train'
    df_train_scores['eta'] = eta

    df_test_scores['stage'] = 'test'
    df_test_scores['eta'] = eta

    # combine scores
    df_bdd_scores = pd.concat([df_train_scores, df_test_scores]).reset_index(drop=True)
    df_bdd_scores['thld'] = thld_name


    # save data and scores
    if save_path is not None:
        temp_name = thld_name
        if neurons is not None:
            temp_name += "-neurons"
        # some bdds are too big to save
        # thus the processed data and bdd results will be saved instead
        # with open(save_path / f'{temp_name}.pkl', "wb") as f:
        #     pickle.dump(patterns, f, pickle.HIGHEST_PROTOCOL)

        # save scores
        df_bdd_info.to_csv(save_path / f'info-{temp_name}.csv', index=False)
        df_bdd_scores.to_csv(save_path / f'scores-{temp_name}.csv', index=False)

        # save processed data and the BDD result
        df_train.to_csv(save_path / f'{temp_name}_bdd_signle_eta_train.csv', index=False)
        df_test.to_csv(save_path / f'{temp_name}_bdd_signle_eta_test.csv', index=False)


    # delete variables
    del BDD, patterns
    del df_train_scores, df_test_scores
    gc.collect()

    print(f'> Done! [ {thld_name} - eta: {eta} ]')

    return df_bdd_info, df_bdd_scores