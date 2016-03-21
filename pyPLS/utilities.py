import numpy as np
import pandas as pd
from .preprocessing import scaling
def nanmatprod(ma, mb):
    """
    Compute matrix product of a and b in cas there is missing values.
    :type ma: a numpy array or matrix
    :type mb: a numpy array or matrix
    :return: a numpy array or matrix
    """

    ma_was_matrix = False
    if type(ma) == np.matrix:
        ma = np.array(ma)
        ma_was_matrix = True

    mb_was_matrix = False
    if type(mb) == np.matrix:
        mb = np.array(mb)
        mb_was_matrix = True

    nra = ma.shape[0]
    try:
        nca = ma.shape[1]
    except IndexError:
        nca = 1

    nrb = mb.shape[0]
    try:
        ncb = mb.shape[1]
    except IndexError:
        ncb = 1
    mc = np.zeros((nra, ncb))

    if nca == nrb:
        if nra > 1:
            if ncb > 1:
                for i in np.arange(nra):
                    mc[i, :] = np.nansum(mb.T * ma[i, :], 1)
            else:
                for i in np.arange(nra):
                    mc[i, :] = np.nansum(mb.T * ma[i, :])
        else:
            mc[0, :] = np.nansum(mb.T * ma, 1)
    else:
        raise ValueError("shapes " + str(ma.shape) + " and " + str(mb.shape) + " are not aligned")

    if ma_was_matrix and mb_was_matrix:
        mc = np.matrix(mc)

    return mc


def ROCanalysis(Yhat, classes, positive, npoints=50):
    """

    :param Yhat:
    :param classes:
    :param positive:
    :param npoints:
    :return:
    """
    if isinstance(positive, str):
            positive = [positive]
    if isinstance(positive, int):
            positive = [str(positive)]

    classes = [str(di) for di in classes]
    minLim = min(Yhat)
    maxLim = max(Yhat)
    limit = np.arange(minLim, maxLim, (maxLim-minLim)/npoints)
    falsePositive = np.zeros(len(limit))
    falseNegative = np.zeros(len(limit))
    truePositive = np.zeros(len(limit))
    trueNegative = np.zeros(len(limit))

    for k, lim in enumerate(limit):
        for i, y in enumerate(Yhat):
            if y < lim:
                if classes[i] in positive:
                    falseNegative[k] += 1
                else:
                    trueNegative[k] += 1

            else:
                if classes[i] not in positive:
                    falsePositive[k] += 1
                else:
                    truePositive[k] += 1

    sensitivity = truePositive / (truePositive + falseNegative)
    specificity = trueNegative / (trueNegative + falsePositive)

    return specificity, sensitivity


def isValid(X, forPrediction=False):
    """
    Check validity of input for PLS and PCA
    :param X: an array like
    :return: the validated numpy array
    """
    # If X is a Pandas DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.as_matrix()
        if X.dtype == object:
            X = np.asarray(X, dtype=np.float64)

    if type(X) == pd.Series:
        X = X.as_matrix()
        if X.dtype == object:
            X = np.asarray(X, dtype=np.float64)

    if isinstance(X, list):
        try:
            X = np.asarray(X, dtype=np.float64)
        except ValueError:
            raise ValueError("Cannot cast X into an array of floats")

    if X.ndim > 2:
        raise ValueError("Y cannot have more than 2 dimensions")

    if X.ndim < 2:
        if forPrediction is True:
            X = np.expand_dims(X, axis=0)
        else:
            X = np.expand_dims(X, axis=1)

    if isinstance(X, np.ndarray):
        try:
            n, p = X.shape
        except ValueError:
            raise ValueError("X must be a 2D numpy array")

    else:
        raise ValueError("X must be an array like object")

    return X, n, p

def corr(X,Y):
    X, nx, px = isValid(X)
    Y, ny, py = isValid(Y)
    X, Xbar, Sx = scaling(X,1)
    Y, Ybar, Sy = scaling(Y,1)

    if nx == ny:
        return X.T @ Y / (nx - 1)
    else:
        raise ValueError("X and Y must have the same number of rows.")



if __name__ == '__main__':
    a = np.random.randn(30, 20)
    a[0,6] = np.nan
    b = np.random.randn(20,6)
    print(nanmatprod(a, b))
