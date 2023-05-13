import multiprocessing as mp
from itertools import product
import pandas as pd

def add_row_df(comb):
    df = pd.DataFrame({'comb': []})
    df.loc[df.shape[0]+1] = [comb]
    return df

p = [0, 1, 2, 3, 4]
eta = [0, 1, 2, 3]

combinations = [p for p in product(p, eta)]


print('Numper of available CPUs:', mp.cpu_count())
print('Numper combinations:', len(combinations))

pool = mp.Pool(min( len(combinations), mp.cpu_count()))

results = pool.map(add_row_df, combinations)

pool.close()

print(pd.concat(results).reset_index(drop=True))