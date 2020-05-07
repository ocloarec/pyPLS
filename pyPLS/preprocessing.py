import numpy as np


def diagonal_correction(K, v):

    n = K.shape[0]
    correction = np.zeros((n, n))
    H = (np.eye(n) - np.ones((n, n), dtype=float)/n)
    for i, k in enumerate(K):
        kw = np.delete(k, i)
        kwc = kw - np.mean(kw)
        vw = np.delete(v, i)
        vwc = vw - np.mean(vw)
        a = np.inner(kwc, vw)/np.sum(vwc**2)
        b = np.mean(kw) - a * np.mean(vw)
        kihat = v[i]*a + b
        correction[i, i] = k[i] - kihat

    K = K - correction
    # ZZ is then recentred
    K = H.T @ K @ H
    return K


def scaling(X, scale, center=True, axis=0):
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

    if center:
        X = X - Xbar

    X = X / (Sx**scale)
    if was_matrix:
        X = np.matrix(X)

    return X, Xbar, Sx
