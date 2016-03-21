import numpy as np
from scipy.spatial.distance import pdist, squareform
from . import utilities


def linear(X, **kwargs):
    """

    Parameters
    ----------
    X
    Y (optional) default is None

    Returns
    -------
    The linear kernel
    """
    Y = None
    for key, value in kwargs.items():
        if key == "Y":
            Y = value

    # Validation of X and Y
    X, nx, px = utilities.isValid(X)

    if Y is None:
        Y = X
    else:
        Y, ny, py = utilities.isValid(Y)

    if np.isnan(X).any() or np.isnan(X).any():
        return utilities.nanmatprod(X, Y.T)
    else:
        return X @ Y.T


def gaussian(X, **kwargs):
    """

    Parameters
    ----------
    X : the core matrix
    sigma (optional) default is 1.0

    Returns
    -------
    The gaussian kernel
    """

    Y = None
    sigma = None
    for key, value in kwargs.items():
        if key == "Y":
            Y = value
        if key == "sigma":
            sigma = value

    # Validation of X, Y and sigma
    X, nx, px = utilities.isValid(X)

    if Y is None:
        Y = X
        ny = nx
    else:
        Y, ny, py = utilities.isValid(Y)

    if sigma is None:
        sigma = 1
    else:
        assert float(sigma), "sigma must be a number"

    # Kernal calculation
    if np.isnan(X).any() or np.isnan(Y).any():
        K = np.zeros((nx, ny))
        if nx == ny:
            for i in np.arange(nx):
                for j in range(i):
                    K[i, j] = np.nansum(np.square(X[i, :] - X[j, :]))
                    K[j, i] = K[i, j]
        else:
            for i in np.arange(nx):
                for j in range(ny):
                    K[i, j] = np.nansum(np.square(X[i, :] - X[j, :]))

    else:
        if Y is not None:
            X = np.concatenate((X, Y))
            K = squareform(pdist(X))[nx:, :nx].T  # TODO: Check this!
        else:
            K = squareform(pdist(X)).T

    return np.exp(-K/sigma)

IMPLEMENTED_KERNEL = {"linear": linear, "gaussian": gaussian}


if __name__ == '__main__':
    from .simulation import simulateData
    Xt, Z, Yt = simulateData(50, 5, 1000, 10., signalToNoise=100.)
