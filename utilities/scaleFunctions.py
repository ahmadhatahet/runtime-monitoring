from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, concat


def fitStandardScalerSingle(df, numNeurons):
    """
    Fit StandardScaler to dataframe and ignore columns after numNeurons
    Return: scaler object"""

    scaler = StandardScaler()
    scaler.fit(df.iloc[:, :numNeurons])

    return scaler


def fitScalerClasses(df, classesColumn, numNeurons):
    """
    Fit StandardScaler to dataframe and ignore columns after numNeurons
    Return: scaler object"""

    scalers = {}
    classes = df[classesColumn].drop_duplicates().sort_values().tolist()
    # fit a scaler class wise
    for c in classes:
        scaler = StandardScaler()
        scaler.fit(df.loc[df[classesColumn] == c, df.columns[:numNeurons]])
        scalers[c] = scaler

        del scaler

    return scalers



def applyStandardScalerSingle(scaler, df):
    """Apply scaler to dataframe"""
    numNeurons = scaler.n_features_in_
    dfScaler = DataFrame( scaler.transform(df.iloc[:, :numNeurons]), columns=df.columns[:numNeurons] )
    # readd non neuron columns
    for col in df.columns[numNeurons:].tolist():
        dfScaler[col] = df[col].to_numpy()

    return dfScaler



def applyScalerClasses(scalers, df, classesColumn):
    """Apply scaler to dataframe class wise"""
    dfClassesScaler = DataFrame()
    classes = df[classesColumn].drop_duplicates().sort_values().tolist()
    # for each class
    for c in classes:
        scaler = scalers[c]
        numNeurons = scaler.n_features_in_
        # apply scaler
        dfScaler = DataFrame( scaler.transform(df.loc[df[classesColumn] == c, df.columns[:numNeurons]]), columns=df.columns[:numNeurons] )
        # readd non neuron columns
        for col in df.columns[numNeurons:].tolist():
            dfScaler[col] = df.loc[df[classesColumn] == c, col].to_numpy()
        # conactenate with finished data
        dfClassesScaler = concat([dfClassesScaler, dfScaler])

        del dfScaler, scaler

    return dfClassesScaler

