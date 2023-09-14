import numpy as np
import pandas as pd
from datetime import datetime as dt
from dd import cudd
import time
import gc
from pathlib import Path
from utilities.utils import save_pickle


class MonitorBDD:
    def __init__(self, num_neurons, thld, neurons=[], reorder=False, memory=10):
        # create instance of bdd class
        self.bdd = cudd.BDD(memory_estimate= memory * 2**30)
        # turn off automatic reordering
        self.bdd.configure(reordering=False)
        self.reorder = reorder
        # initiate empty bdd
        self.roots = self.bdd.false
        # save threshold
        self.thld = thld
        # number of last hidden layer neurons
        self.num_neurons = num_neurons

        self.neurons = np.array(neurons)
        if self.neurons.shape[0] != 0:
            self.neurons = np.array([int(n[1:]) for n in neurons])
            self.thld = self.thld[self.neurons]

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

        # generate vars either x0, x1, ... per neuron
        if self.neurons.shape[0] != 0: vars_range = self.neurons
        else: vars_range = range(self.num_neurons)

        vars = [f'x{n}' for n in vars_range]

        # add vars to bdd
        for v in vars: self.bdd.add_var(v)

        # generate negative vars
        vars = np.array([ *map(self.bdd.var, vars) ])
        vars_not = np.array([ ~v for v in vars ])

        return vars, vars_not

    def applying_thlds(self, df):
        """TODO"""

        if self.neurons.shape[0] != 0:
            df = df[df.columns[self.neurons]]
        else:
            df = df[df.columns[:self.num_neurons]]

        df_thld = (df >=  self.thld).astype('int8')

        return df_thld.to_numpy()


    def construct_pattern(self, row):
        """Replace 1s with positive variable and 0s with not variables"""
        expr = np.where( row == 1, self.vars, self.vars_not )
        return np.bitwise_and.reduce( expr )


    def __add_one_pattern(self, row):
        """Construct expression from the row and add it to the BDD"""
        self.roots |= self.construct_pattern(row)


    def flip_pattern(self, patterns, eta):
        """flip n-th bit to allow more freedom(false positive)
        if eta = 0 then pattern as is
        if eta = 1 then loop over each bit and force it to one
        eta = 2 loop over 2 bits and flip them ... etc
        drop any duplicate patterns"""

        if eta == 0: yield patterns

        for nth in range(patterns.shape[1]-eta+1):
            t1 = patterns.copy()
            t0 = patterns.copy()

            t1[:, nth:nth+eta] = 1
            t0[:, nth:nth+eta] = 0

            # compare hamming distance to original
            idx1 = (patterns ^ t1).sum(1) == eta
            idx0 = (patterns ^ t0).sum(1) == eta

            if idx1.sum() == 0 and idx0.sum() == 0: continue

            # pick rows with respect to eta
            yield np.vstack([np.unique(t1[idx1], axis=0), np.unique(t0[idx0], axis=0)])


    def add_dataframe(self, df, eta=0, eval_dfs=[]):
        """TODO"""

        # to formalize some calculations
        build_time = lambda start: round(int(time.perf_counter() - start) / 60, 3)
        date_to_str = lambda date: dt.strftime(date, '%Y-%m-%d %H:%M:%S')

        patterns = self.applying_thlds(df)

        # only unique patterns
        patterns = np.unique(patterns, axis=0)


        for et in range(eta+1):
            row = self.stats.shape[0]+1
            self.stats.loc[row, 'start_time'] = date_to_str(dt.now())

            start = time.perf_counter()

            if et == 0:
                num_pattern = df.shape[0]
                num_added_pattern = patterns.shape[0]
                for i in range(patterns.shape[0]):
                    self.__add_one_pattern(patterns[i])

            else:
                # generate flipped patterns for an eta
                num_pattern = 0
                num_added_pattern = 0
                for flipped_patterns in self.flip_pattern(patterns, et):
                    # add number of all
                    num_pattern += flipped_patterns.shape[0]
                    bdd_recognized = np.apply_along_axis(self.__check_one_pattern, 1, flipped_patterns).astype(bool)
                    num_added_pattern += np.sum(~bdd_recognized)
                    flipped_patterns = flipped_patterns[bdd_recognized]
                    for pattern in flipped_patterns:
                        self.__add_one_pattern(pattern)

            # manual call for reorder
            if self.reorder:
                cudd.reorder(self.bdd)

            bdd_stats = self.bdd.statistics()
            self.stats.loc[row, 'eta'] = et
            self.stats.loc[row, 'build_time_min'] = build_time(start)
            self.stats.loc[row, 'size_mb'] = round( self.bdd.statistics()['mem'] * 1e-6, 1)
            self.stats.loc[row, 'reorder_time_min'] = round(bdd_stats['reordering_time'] / 60 , 3)
            self.stats.loc[row, 'num_reorder'] = bdd_stats['n_reorderings']
            self.stats.loc[row, 'num_neurons'] = len(self.neurons) if self.neurons.shape[0] != 0 else self.num_neurons
            self.stats.loc[row, 'end_time'] = date_to_str(dt.now())
            self.stats.loc[row, 'num_patterns'] = num_pattern
            self.stats.loc[row, 'num_unique_patterns_%'] = round(num_added_pattern / max(num_pattern, 1) * 100, 1)

            # add column for scoring
            if len(eval_dfs) != 0:
                for eval_df in eval_dfs:
                    self.evaluate_dataframe(eval_df, et)

        # return evaluated dataframes
        if len(eval_dfs) != 0: return eval_dfs

        return


    def __check_one_pattern(self, row):
        """convert a pattern to expression and run it against bdd"""
        if (self.roots & self.construct_pattern(row) ) == self.bdd.false:
            return 0 # means not found
        else:
            return 1 # found it

    def check_one_pattern(self, row):
        """For testing apply threshold then convert a pattern to expression and run it against bdd"""
        row = (row >= self.thld).astype(np.int8)
        if (self.roots & self.construct_pattern(row) ) == self.bdd.false:
            return 0 # means not found
        else:
            return 1 # found it


    def evaluate_dataframe(self, df, eta=None):
        """check all rows against the BDD
        return the results as a new column with respect to degree of freedom"""
        bdd_results = np.zeros(df.shape[0], dtype=np.int8)

        patterns = self.applying_thlds(df)

        bdd_results = np.apply_along_axis(self.__check_one_pattern, 1, patterns)

        if eta is not None:
            df[f'bdd_{eta}'] = bdd_results
            return

        # if the function called specifically, return scored df after evaluating
        df['bdd'] = bdd_results
        return self.score_dataframe(df)

    def score_dataframe(self, df, bdd_col='bdd'):
        """TODO"""
        # return empty df if false column name was passed

        # how many instances per class
        df_all_classes = df[['y', 'true']].groupby('y').count().sort_index()
        df_all_classes.columns = ['total_count']

        # how many misclassification
        df_misclassified_images = df.loc[df['true'] == False, ['y', bdd_col]].groupby('y').count().sort_index()
        df_misclassified_images.columns = ['total_misclassified']

        # how many patterns not found
        df_unrecognized_images = df.loc[df[bdd_col] == 0, ['y', bdd_col]].groupby('y').count().sort_index()
        df_unrecognized_images.columns = [bdd_col + '_unrecognized']

        # how many patterns not found and misclassified
        df_unrecognized_misclassified_images = df.loc[(df[bdd_col] == 0) & (df['true'] == False), ['y', bdd_col]].groupby('y').count().sort_index()
        df_unrecognized_misclassified_images.columns = [bdd_col + '_unrecognized_and_misclassified']

        # how many patterns not found and misclassified
        df_unrecognized_classified_images = df.loc[(df[bdd_col] == 0) & (df['true'] == True), ['y', bdd_col]].groupby('y').count().sort_index()
        df_unrecognized_classified_images.columns = [bdd_col + '_unrecognized_and_classified']

        df_scores = df_all_classes.join(df_unrecognized_images)\
            .join(df_misclassified_images) \
            .join(df_unrecognized_misclassified_images) \
            .join(df_unrecognized_classified_images) \

        del df_unrecognized_images, df_unrecognized_misclassified_images, df_unrecognized_classified_images

        # add row all
        total_images = df_all_classes['total_count'].sum()
        misclassified_images = df_scores['total_misclassified'].sum()
        unrecognized_images = df_scores[bdd_col + '_unrecognized'].sum()
        unrecognized_and_misclassified_images = df_scores[bdd_col + '_unrecognized_and_misclassified'].sum()
        unrecognized_and_classified_images = df_scores[bdd_col + '_unrecognized_and_classified'].sum()
        df_scores.loc['all', :] = [
            total_images,
            unrecognized_images,
            misclassified_images,
            unrecognized_and_misclassified_images,
            unrecognized_and_classified_images
        ]

        # if data frame return 0 rows, a nan will be placed
        df_scores.fillna(0, inplace=True)

        ## calculate metrics
        # Not recognized patterns
        df_scores['NPR'] = df_scores[bdd_col + '_unrecognized'] / df_scores['total_count']

        # more misclassfied and undetected pattern means the monitor is correctly detecting unfamiliar patterns
        df_scores['NPV'] = df_scores[bdd_col + '_unrecognized_and_misclassified'] / df_scores[bdd_col + '_unrecognized']

        # more unrecognized and missclassified mean the BDD is memorizing the model.
        df_scores['specificity'] = df_scores[bdd_col + '_unrecognized_and_misclassified'] / df_scores['total_misclassified']

        # if class is never Misclassified and bdd recognize all of its patterns
        # both NPR and unrecognized_misclassified will be 0
        # so the division will result in NaN, thus will be replaced by zero
        # because we don't know how the monitor will react once the class's data start to get outdated
        df_scores['NPR'].replace({np.nan:0.0}, inplace=True)
        df_scores['NPV'].replace({np.nan:0.0}, inplace=True)
        df_scores['specificity'].replace({np.nan:0.0}, inplace=True)

        # no missclassification for a class
        df_scores[bdd_col + '_unrecognized'].replace({np.nan:0.0}, inplace=True)
        df_scores[bdd_col + '_unrecognized_and_misclassified'].replace({np.nan:0.0}, inplace=True)
        df_scores[bdd_col + '_unrecognized_and_classified'].replace({np.nan:0.0}, inplace=True)

        # reorder columns
        df_scores = df_scores[['total_count', 'total_misclassified', bdd_col + '_unrecognized',
                               bdd_col + '_unrecognized_and_misclassified',
                               bdd_col + '_unrecognized_and_classified',
                               'NPR','NPV',
                              'specificity']]

        if bdd_col=='bdd':
            return df_scores.reset_index()
        return df_scores


    def score_dataframe_multi_eta(self, df, eta, subset_name=None):
        """TODO"""
        df_scores = pd.DataFrame()

        for et in range(eta+1):
            temp = self.score_dataframe(df, f'bdd_{et}')
            temp['eta'] = et
            temp.columns = [*map(lambda x: x.replace(f'bdd_{et}_', ''), temp.columns)]
            df_scores = pd.concat([df_scores, temp])
            del temp

        df_scores['subset_name'] = subset_name
        df_scores['num_neurons'] = self.stats['num_neurons'].drop_duplicates().values[0]

        return df_scores.reset_index()


def build_bdd_multi_etas(args):
    df_train, df_test, df_true, df_logits, subset_name, neurons, thld_name, thld, eta, memory, save_path = args

    # construcr MonitorBDD
    patterns = MonitorBDD( df_true.shape[1]-1, thld, neurons, False, memory=memory)
    print(f'+ Threshold: {subset_name} - {thld_name}, eta: {eta}')

    # build
    patterns.add_dataframe( df_true, eta, eval_dfs=[df_train, df_test, df_logits] )

    # collect scores
    df_bdd_info = patterns.stats.copy()
    df_bdd_info['thld'] = thld_name
    df_bdd_info['subset_name'] = subset_name

    df_train_scores = patterns.score_dataframe_multi_eta(df_train, eta, subset_name)
    df_test_scores = patterns.score_dataframe_multi_eta(df_test, eta, subset_name)
    df_logit_scores = patterns.score_dataframe_multi_eta(df_logits, eta, subset_name)
    df_train_scores['stage'] = 'train'
    df_test_scores['stage'] = 'test'
    df_logit_scores['stage'] = 'evaluation'

    # combine scores
    df_bdd_scores = pd.concat([df_train_scores, df_test_scores, df_logit_scores]).reset_index(drop=True)
    df_bdd_scores['thld'] = thld_name

    # save data and scores
    if save_path is not None:
        temp_name = thld_name

        if len(neurons) > 0:
            temp_name += f"-{subset_name}"

        # # some bdds are too big to save
        # # thus the processed data and bdd results will be saved instead
        # temp_path = Path('/tmp/ah19') / save_path.parent.name / save_path.name
        # temp_path.mkdir(parents=True, exist_ok=True)

        # filename_ = lambda s, e: f'{s}-{thld_name}-{eta}-{temp_name}.{e}'

        # # multiprocessing can not save serialized objects
        # save_pickle(temp_path / filename_('bdd', 'pkl'), patterns)

        # save scores
        # df_bdd_info.to_csv(save_path / filename_('info', 'csv'), index=False)
        # df_bdd_scores.to_csv(save_path / filename_('score', 'csv'), index=False)

        # # apply threshold
        # if patterns.neurons.shape[0] != 0:
        #     idx_col_keep = patterns.neurons
        #     idx_col_delete = np.delete(df_train.columns[:patterns.num_neurons], idx_col_keep)
        # else:
        #     idx_col_keep = np.arange(patterns.num_neurons)
        #     idx_col_delete = np.delete(df_train.columns[:patterns.num_neurons], idx_col_keep)


        # df_train[df_train.columns[idx_col_keep]] = patterns.applying_thlds(df_train)
        # df_train.drop(idx_col_delete, axis=1, inplace=True)

        # df_test[df_test.columns[idx_col_keep]] = patterns.applying_thlds(df_test)
        # df_test.drop(idx_col_delete, axis=1, inplace=True)

        # df_logits[df_logits.columns[idx_col_keep]] = patterns.applying_thlds(df_logits)
        # df_logits.drop(idx_col_delete, axis=1, inplace=True)

        # # save processed data and the BDD result
        # df_train.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_train.csv', index=False)
        # df_test.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_test.csv', index=False)
        # df_logits.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_evaluation.csv', index=False)

    # delete variables
    del patterns
    del df_train_scores, df_test_scores
    gc.collect()

    print(f'> Done! [ {subset_name} - {thld_name} - eta: {eta} ]')

    return df_bdd_info, df_bdd_scores


def build_bdd_multi_etas_new(args):
    df_train, df_test, df_true, df_logits, dataset, postfix, flavor, subset_name, neurons, thld_name, thld, eta, full, memory, save_path = args

    # output file name
    filename_postfix = f"{dataset}_{postfix}"
    POSTFIX2 = flavor.lower()
    POSTFIX2 += f"_{filename_postfix}"

    if full: fn_temp_name = lambda x: f'all-thlds-full-{x}-{eta}-{subset_name}-{thld_name}-{POSTFIX2}.csv'
    else: fn_temp_name = lambda x: f'all-thlds-{x}-{eta}-{subset_name}-{thld_name}-{POSTFIX2}.csv'

    if (save_path / fn_temp_name('info')).is_file():
        return

    # construcr MonitorBDD
    patterns = MonitorBDD( df_true.shape[1]-1, thld, neurons, False, memory=memory)
    print(f'+ Threshold: {subset_name} - {thld_name}, eta: {eta}')

    # build
    patterns.add_dataframe( df_true, eta, eval_dfs=[df_train, df_test, df_logits] )

    # collect scores
    df_bdd_info = patterns.stats.copy()
    df_bdd_info['thld'] = thld_name
    df_bdd_info['subset_name'] = subset_name

    df_train_scores = patterns.score_dataframe_multi_eta(df_train, eta, subset_name)
    df_test_scores = patterns.score_dataframe_multi_eta(df_test, eta, subset_name)
    df_logit_scores = patterns.score_dataframe_multi_eta(df_logits, eta, subset_name)
    df_train_scores['stage'] = 'train'
    df_test_scores['stage'] = 'test'
    df_logit_scores['stage'] = 'evaluation'

    # combine scores
    df_bdd_scores = pd.concat([df_train_scores, df_test_scores, df_logit_scores]).reset_index(drop=True)
    df_bdd_scores['thld'] = thld_name

    # save data and scores
    df_bdd_info.to_csv(save_path / fn_temp_name('info'), index=False)
    df_bdd_scores.to_csv(save_path / fn_temp_name('scores'), index=False)

        # # some bdds are too big to save
        # # thus the processed data and bdd results will be saved instead
        # temp_path = Path('/tmp/ah19') / save_path.parent.name / save_path.name
        # temp_path.mkdir(parents=True, exist_ok=True)

        # filename_ = lambda s, e: f'{s}-{thld_name}-{eta}-{temp_name}.{e}'

        # # multiprocessing can not save serialized objects
        # save_pickle(temp_path / filename_('bdd', 'pkl'), patterns)

        # save scores
        # df_bdd_info.to_csv(save_path / filename_('info', 'csv'), index=False)
        # df_bdd_scores.to_csv(save_path / filename_('score', 'csv'), index=False)

        # # apply threshold
        # if patterns.neurons.shape[0] != 0:
        #     idx_col_keep = patterns.neurons
        #     idx_col_delete = np.delete(df_train.columns[:patterns.num_neurons], idx_col_keep)
        # else:
        #     idx_col_keep = np.arange(patterns.num_neurons)
        #     idx_col_delete = np.delete(df_train.columns[:patterns.num_neurons], idx_col_keep)


        # df_train[df_train.columns[idx_col_keep]] = patterns.applying_thlds(df_train)
        # df_train.drop(idx_col_delete, axis=1, inplace=True)

        # df_test[df_test.columns[idx_col_keep]] = patterns.applying_thlds(df_test)
        # df_test.drop(idx_col_delete, axis=1, inplace=True)

        # df_logits[df_logits.columns[idx_col_keep]] = patterns.applying_thlds(df_logits)
        # df_logits.drop(idx_col_delete, axis=1, inplace=True)

        # # save processed data and the BDD result
        # df_train.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_train.csv', index=False)
        # df_test.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_test.csv', index=False)
        # df_logits.to_csv(save_path / f'data-{thld_name}-{eta}-{temp_name}_bdd_evaluation.csv', index=False)

    # delete variables
    del patterns
    del df_train_scores, df_test_scores
    gc.collect()

    print(f'> Done! [ {subset_name} - {thld_name} - eta: {eta} ]')

    return subset_name, thld_name, df_bdd_info, df_bdd_scores



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