import numpy as np
import pandas as pd

from pyPLS.pca import prcomp


def unit(X):
    """
    :param X: Matrix to be noemalised
    :return: a normalised matrix, factor used for normalisation, reference vector
    """

    p = X.shape[1]
    factors = np.sum(X, axis=1)
    X = X / np.tile(factors, (p, 1)).T

    return X, factors


def pqn(X, reference=None, sigma=1e-14):

    # Check for DataFrame
    isDataFrame = False
    idx = None
    columns = None
    if isinstance(X, pd.DataFrame):
        # save dataframe columns and indices
        isDataFrame = True
        columns = X.columns
        idx = X.index
        X = X.as_matrix()


    n = X.shape[0]

    if reference is None:
        pc = prcomp(X, 2, scaling=0)
        refIndex = np.argmin(np.sqrt(pc.scores[0]**2 + pc.scores[1]**2))
        reference = X[refIndex, :]

    # Strip the noise or zero values
    mask = reference > sigma
    Quot = X[:, mask]/np.tile(reference[mask], (n, 1))
    factors = np.median(Quot, axis=1)

    if isDataFrame:
        X = pd.DataFrame(X, index=idx, columns=columns)
    return X, factors, reference


if __name__ == '__main__':
    Xtest = np.random.randn(20, 100)+5.0
    # unit normalisation
    Xunit = unit(Xtest)
    # PQN normalisation
    Xpqn = pqn(Xtest, Xtest[3,:])

    # Test with
    Xpd = pd.DataFrame(Xtest)
    Xpqn, factors, reference = pqn(Xpd)
    print(factors)


