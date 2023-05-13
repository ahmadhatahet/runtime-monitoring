from sklearn.decomposition import PCA
from pandas import DataFrame, concat
import numpy as np


def fitPCAClasses(df, classesColumn, scalers={}, numNeurons=None):

    pcas = {}
    classes = df[classesColumn].drop_duplicates().sort_values().tolist()

    for c in classes:

        if scalers != {}: numNeurons = scalers[c].n_features_in_
        pcas[c] = PCA(n_components=numNeurons)
        temp = df.loc[df[classesColumn] == c, df.columns[:numNeurons]]
        if scalers != {}:
            temp = scalers[c].transform(df.loc[df[classesColumn] == c, df.columns[:numNeurons]])
        pcas[c].fit(temp)

        del temp

    return pcas


def applyPCAClasses(pcas, df, classesColumn, scalers={}):

    dfPCA = DataFrame()
    classes = df[classesColumn].drop_duplicates().sort_values().tolist()

    for c in classes:

        numNeurons = pcas[c].n_components
        temp = df.loc[df[classesColumn] == c, df.columns[:numNeurons]]
        if scalers != {}:
            temp = scalers[c].transform(temp)
        temp = DataFrame( pcas[c].transform(temp), columns=[f'PC{i}' for i in range(1, numNeurons+1)] )

        for col in df.columns[numNeurons:].tolist():
            temp[col] = df.loc[df[classesColumn] == c, col].to_numpy()

        dfPCA = concat([dfPCA, temp])

        del temp

    return dfPCA


def neuronsLoadingsClasses(pcas, numNeurons, var_thld=0.9, loadings_thld=0.5):

    neurons = {}

    for cls, pca in pcas.items():

        num_component = sum(np.cumsum(pca.explained_variance_ratio_).round(2) <= var_thld)
        if numNeurons is None: numNeurons = pca.n_components

        pca_components = DataFrame(
            pca.components_[:,:num_component],
            columns=[f'PC_{i}' for i in range(num_component)],
            index=[f'x{i}' for i in range(numNeurons)]
        )

        pca_loadings = pca_components * np.sqrt(pca.explained_variance_[:num_component])

        neurons_loadings = (pca_loadings.abs() >= loadings_thld).any(axis=1)
        neurons[cls] = neurons_loadings[neurons_loadings == True].index.tolist()

    return neurons


def fitPCASingle(df, scaler=None, numNeurons=None):

    if scaler is not None: numNeurons = scaler.n_features_in_
    pca_ = PCA(n_components=numNeurons)
    if scaler is not None:
        df = scaler.transform(df[df.columns[:numNeurons]])
    pca_.fit(df)

    return pca_


def applyPCASingle(pca, df, scaler=None, numNeurons=None):

    if numNeurons is None: numNeurons = pca.n_components
    temp = df[df.columns[:numNeurons]]
    if scaler is not None:
        temp = scaler.transform(temp)
    temp = DataFrame( pca.transform(temp), columns=[f'PC{i}' for i in range(1, numNeurons+1)] )

    for col in df.columns[numNeurons:].tolist():
        temp[col] = df[col].to_numpy()

    return temp


def numComponents(pca, var_thld=0.9):
    return sum(np.cumsum(pca.explained_variance_ratio_).round(2) <= var_thld)


def neuronsLoadingsSingle(pca, numNeurons=None, var_thld=0.9, loadings_thld=0.5):

    num_component = sum(np.cumsum(pca.explained_variance_ratio_).round(2) <= var_thld)
    if numNeurons is None: numNeurons = pca.n_components

    pca_components = DataFrame(
        pca.components_[:,:num_component],
        columns=[f'PC_{i}' for i in range(num_component)],
        index=[f'x{i}' for i in range(numNeurons)]
    )

    pca_loadings = pca_components * np.sqrt(pca.explained_variance_[:num_component])

    neurons_loadings = (pca_loadings.abs() >= loadings_thld).any(axis=1)

    return neurons_loadings[neurons_loadings == True].index.tolist()

if __name__ == "__main__":
    ...