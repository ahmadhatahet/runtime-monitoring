from dd.autoref import BDD
import numpy as np
from fastprogress import master_bar, progress_bar
import matplotlib.pyplot as plt


class MonitorSingleBDD:
    def __init__(self, num_classes, num_neurons, thld_1, thld_2=None, thld_3=None, subset_neurons=None, verbose=False):

        self.bdd = BDD()
        self.roots = self.bdd.false
        self.class_expr = {k: np.unpackbits( np.array([k], dtype=np.uint8) )
                           for k in range(num_classes)}
        self.numClassBits = 8
        self.num_classes = num_classes
        self.num_neurons = num_neurons
        self.num_vars = num_neurons
        self.verbose = verbose

        self.thld_1 = thld_1
        self.thld_2 = thld_2
        self.thld_3 = thld_3
        self.num_bits = 2 if thld_2 is not None or thld_3 is not None else 1

        if subset_neurons is not None:
            self.subset_neurons = {int(k): np.array(v) for k, v in subset_neurons.items()}
        else:
            self.subset_neurons = None

        self.vars, self.vars_not = self.declare_vars()


    def declare_vars(self):

        # each class will be repr in 8 bits
        class_vars = [f'x{i}_{0}' for i in range(self.numClassBits)]
        [ *map(self.bdd.add_var, class_vars) ]
        vars = [ *map(self.bdd.var, class_vars) ]

        for n in range(self.numClassBits, self.num_neurons + self.numClassBits):
            # generate vars either x0_0 or x0_0 and x0_1 per neuron
            v = [f'x{n}_{i}' for i in range(self.num_bits)]

            # add vars to bdd
            [ *map(self.bdd.add_var, v) ]

            # append vars to variable
            vars.extend( [ *map(self.bdd.var, v) ] )

        # generate negative vars
        vars = np.array(vars)
        vars_not = np.array([ ~v for v in vars ])

        return vars, vars_not


    def construct_pattern(self, cls, row):

        thld_1 = self.thld_1[cls]
        vars = self.vars
        vars_not = self.vars_not
        if self.thld_2 is not None:
            thld_2 = self.thld_2[cls]
        if self.thld_3 is not None:
            thld_3 = self.thld_3[cls]

        if self.subset_neurons is not None:
            row = self.select_neurons(cls, row)
            thld_1 = thld_1[self.subset_neurons[cls]]
            vars = np.append(vars[:self.numClassBits], vars[self.subset_neurons[cls] + self.numClassBits])
            vars_not = np.append(vars_not[:self.numClassBits], vars_not[self.subset_neurons[cls] + self.numClassBits])
            if self.num_bits == 2:
                thld_2 = thld_2[self.subset_neurons[cls]]
                thld_3 = thld_3[self.subset_neurons[cls]]

        row = row.to_numpy()


        pattern_row = np.greater(row, thld_1).astype(int)

        # add second bit
        if self.thld_2 is not None:
            pattern_row += np.greater(row, thld_2).astype(int)

        # add second bit
        if self.thld_3 is not None:
            pattern_row += np.greater(row, thld_3).astype(int)


        if self.num_bits == 2:
            pattern_row = pattern_row.reshape(row.shape[0], 1)

            pattern_row = np.where(
                pattern_row == [0], [0, 0], # 0
                np.where(
                    pattern_row == [1], [1, 0], # 1
                    np.where(
                        pattern_row == [2], [0, 1], # 2
                        [1, 1] # 3
                        )
                    )
                )

            pattern_row = np.reshape(pattern_row, pattern_row.shape[0] * 2)

        # replace 1 with vars and 0 with vars_not
        pattern_row = np.append( self.class_expr[cls] , pattern_row )
        expr = np.where( pattern_row == 1, vars, vars_not )
        expr = np.bitwise_and.reduce( expr )

        return expr


    def select_neurons(self, cls, row):
        return row[ self.subset_neurons[cls] ]


    def check_pattern_length(self, row):

        if self.num_bits == 2:
            assert len(self.vars)/2 == row.shape[0], "ERROR: VARS and ROW do not match!"
        else:
            assert len(self.vars) == row.shape[0], "ERROR: VARS and ROW do not match!"


    def add_one_pattern(self, cls, row):
        # add pattern to bdd
        expr = self.construct_pattern(cls, row)
        self.roots |= expr


    def add_multi_pattern(self, df, cls, mb=None):
        pb = progress_bar(range(df.shape[0]), parent=mb)

        for _, (_, row) in zip(pb, df.iterrows()):

            self.add_one_pattern(cls, row)


    def add_dataframe(self, df, cls=None):

        if cls is None:
            cls = df['y'].drop_duplicates().sort_values().values

        num_classes = len(cls)
        mb = master_bar(range(num_classes))

        for _, cls in zip(mb, cls):

            if self.verbose: mb.write(f"Building expression for class: {cls}")

            df_cls = df[ df['y'] == cls ].drop('y', axis=1)
            self.add_multi_pattern(df_cls.drop_duplicates(), cls, mb)



    def check_one_pattern(self, cls, row):


        # add pattern to bdd
        expr = self.construct_pattern(cls, row)

        if (self.roots & expr) == self.bdd.false:
            return False # means not found
        else:
            return True # found it


    def check_multi_pattern(self, df, cls, mb=None):

        result = np.zeros(df.shape[0], dtype=bool)
        pb = progress_bar(range(df.shape[0]), parent=mb)

        for i, (indx, row) in zip(pb, df.iterrows()):
            result[i] = self.check_one_pattern(cls, row)

        return result


    def evaluate_dataframe(self, df, cls=None, cross_classes=False):

        if cls is None:
            cls = df['y'].drop_duplicates().sort_values().values

        if cross_classes:
            confusion_matrix = np.zeros((len(cls),len(cls)))

        mb = master_bar(range(len(cls)))

        if 'bdd' not in df.columns:
            df['bdd'] = False

        ignored_columns = df.columns[self.num_neurons:]

        for _, c in zip(mb, cls):

            if self.verbose: mb.write(f"Building expression for class: {c}")

            df_cls = df[ df['y'] == c ].drop(ignored_columns, axis=1)

            result = self.check_multi_pattern(df_cls, c, mb)

            df.loc[ df['y'] == c, 'bdd' ] = result

            if cross_classes:
                confusion_matrix[c][c] = result.sum()
                pb = progress_bar(range(len(cls)-1), parent=mb)
                for _, second in zip(pb, cls[cls != c]):
                    confusion_matrix[c][second] = self.check_multi_pattern(df_cls, second, mb).sum()


        if cross_classes:
            return df, confusion_matrix

        return df

    def score_dataframe(self, df):

        df['TP'] = (df['bdd'] == True) & (df['true'] == True)
        df['TN'] = (df['bdd'] == False) & (df['true'] == False)
        df['FP'] = (df['bdd'] == True) & (df['true'] == False)
        df['FN'] = (df['bdd'] == False) & (df['true'] == True)

        df_scores = df[['y', 'TP','FN', 'TN','FP']].groupby('y').sum().sort_index()

        df_scores['Accuracy'] = np.divide(df_scores[['TP', 'TN']].sum(axis=1), df_scores.sum(axis=1))
        df_scores['ErrorRate'] = 1 - df_scores['Accuracy']
        df_scores['Precision'] = np.divide(df_scores['TP'] , df_scores[['TP', 'FP']].sum(axis=1))
        df_scores['Specificity'] = np.divide(df_scores['TN'] , df_scores[['FP', 'TN']].sum(axis=1))
        df_scores['Recall'] = np.divide(df_scores['TP'] , df_scores[['TP', 'FN']].sum(axis=1))
        df_scores['F1'] = np.divide(df_scores['TP'] , df_scores[['FP', 'FN']].sum(axis=1) * 0.5 + df_scores['TP'])

        df_scores.fillna(0, inplace=True)

        return df, df_scores


    def plot_stats(self, df, keys, stage, save_folder=None, prefix=None):

        if isinstance(keys, list):
            title = f'{stage} - {keys[0].lower()}: {df[keys[0]].mean().round(2)} / {keys[1].lower()}: {df[keys[1]].mean().round(2)}\n'
            filename = f'{stage.lower()}_{"_".join(map(lambda x: x.lower(), keys))}'
            df[keys].plot(kind='bar', stacked=True, title=title, legend=False, xlabel='', hatch='x', edgecolor='black')
        else:
            title = f'{stage} - {keys.lower()}: {df[keys].mean().round(2)}\n'
            filename = f'{stage.lower()}_{keys.lower()}'
            df[keys].plot(kind='bar', title=title, legend=False, xlabel='', hatch='x', edgecolor='black')

        plt.axhline(df[keys[0]].mean().round(3), color='red', linewidth=2, linestyle='--')


        if save_folder is not None:
            plt.savefig(save_folder / f'monitor_bdd_{filename}_{prefix}.jpg', dpi=150, transparent=False)

        plt.show()


    def save_scores(self, df, path):
        df.to_csv(path, index=False)