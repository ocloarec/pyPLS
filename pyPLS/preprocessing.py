import numpy as np


def scaling(X, scale, axis=0):
    """
    Scale the array using the provided scaling factor

    """

    if X.dtype == object:
        X = np.asarray(X, dtype=np.float64)

    was_matrix = False
    if type(X) == np.matrix:
        X = np.asarray(X)
        was_matrix = True

    # Any missing values?
    missing_value = False
    if np.isnan(X).any():
        missing_value = True

    if missing_value:
        Xbar = np.nanmean(X, axis=axis)
        Sx = np.nanstd(X, axis=axis)
    else:
        Xbar = X.mean(axis=axis)
        Sx = X.std(axis=axis)
    X = X - Xbar

    X = X / (Sx**scale)
    if was_matrix:
        X = np.matrix(X)

    return X, Xbar, Sx
