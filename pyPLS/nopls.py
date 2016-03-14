from __future__ import print_function
import numpy as np
from scipy.spatial.distance import pdist, squareform
from ._PLSbase import _pls
from .utilities import nanmatprod
from .engines import kpls as _kpls

IMPLEMENTED_KERNEL = ["linear", "gaussian"]


# Definition of the supported kernel
def linear(X,Y=None):
    """
    Linear kernel
    :param X:
    :return: the kernel
    """
    if Y is None:
        Y = X
    if np.isnan(X).any() or np.isnan(X).any():
        return nanmatprod(X, Y.T)
    else:
        return X @ Y.T


def gaussian(X, Y=None, sigma=1.0):
    if Y is None:
        Y = X
    nx = X.shape[0]
    ny = Y.shape[0]
    if np.isnan(X).any() or np.isnan(Y).any():
        K = np.zeros((nx,ny))
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
            X = np.concatenate((X,Y))
            K = squareform(pdist(X))[nx:, :nx] #¢ TODO: Check this!
        else:
            K = squareform(pdist(X))

    return np.exp(-K/sigma)




class nopls(_pls):
    """
        This is the multivariate noPLS algorithm.
        Parameters:
            X: {N, P} array like
                a table of N observations (rows) and P variables (columns) - The explanatory variables,
            Y: {N, Q} array like
                a table of N observations (rows) and Q variables (columns) - The dependent variables,
            scaling: float, optional
                A number typically between 0.0 and 1.0 corresponding to the scaling, typical example are
                0.0 corresponds to mean centring (default)
                0.5 corresponds to Pareto scaling
                1.0 corresponds to unit variance scaling
            kernel: Related to kerneL-PLS define how to calculate XX'
                default = linear
            cvfold: int, optional
                the number of folds in the cross-validation - default is 7

        Returns
        -------
        out : a nopls2 object with ncp components
            Attributes:
                ncp: number of components fitted
                T : PLS scores table
                P : PLS loadings table (if linear kernel)
                C : PLS score regression coefficients
                B : PLS regression coefficients
                Yhat: model predicted Y
                Yhatcv: cross-validation predicted Y
                R2Y: Determination coefficients of Y
                Q2Ycol: Cross validation parameters per colums of Y
                Q2Ycum: Cumulative cross validation parameter

            Methods:
                scores(n), loadings(n), weights(n)
                    n: int
                        component id
                    return the scores of the nth component
            predict(Xnew)
                Xnew: array like
                    new observation with the same number of variables tha X
                return predicted Y
    """
    def __init__(self, X, Y, kernel=linear,
                 auto_penalization=True,
                 cvfold=None,
                 scaling=0,
                 ncp=None,
                 err_lim=1e-9,
                 nloop_max=200,
                 sigma=None):

        _pls.__init__(self, X, Y, scaling=scaling)

        self.model = "nopls"
        self.err_lim = err_lim
        self.nloop_max = nloop_max
        assert kernel in IMPLEMENTED_KERNEL, "Kernel not supported!"

        self.K = None

        if kernel == "linear":
            self.K = linear(X)
        if kernel == "gaussian" and sigma is not None:
            self.K = gaussian(X, sigma)

        assert not np.isnan(self.K).any(), "Kernel calculation lead to missing values!"

        if self.missingValuesInY:
            YY = nanmatprod(self.Y, self.Y.T)
        else:
            if self.py > 1:
                YY = self.Y.dot(self.Y.T)
            else:
                YY = np.outer(self.Y, self.Y)

        assert not np.isnan(YY).any(), "A row of Y contains only missing values!"

        #####################
        self.T, self.U, self.C, self.Kcorr, self.warning = _kpls(self.K,
                                                                 self.Y,
                                                                 auto_penalization=auto_penalization,
                                                                 ncp=ncp,
                                                                 err_lim=err_lim,
                                                                 nloop_max=nloop_max)
        #####################
        # Deduction of the number of component fitted from the score array
        self.ncp = self.T.shape[1]

        if kernel == "linear":
            if self.missingValuesInX:
                self.P = nanmatprod(self.X.T, self.T.dot(np.linalg.inv(self.T.T.dot(self.T))))
            else:
                self.P = self.X.T.dot(self.T).dot(np.linalg.inv(self.T.T.dot(self.T)))

            # Regression coefficient and model prediction
            self.B = self.X.T @ self.U @ np.linalg.inv(self.T.T @ self.Kcorr @ self.U) @ self.T.T @ self.Y
            if self.B.ndim < 2:
                self.B = np.expand_dims(self.B, axis=1)
        else:
            self.P = None
            self.B = None

        if self.B:
            self.Yhat = self.X @ self.B
        else:
            self.Yhat = self.T @ self.T.T @ self.Y

        self.R2Y, self.R2Ycol = self._calculateR2Y(self.Y, self.Yhat)
        # self.R2Y_dev, self.R2Y_devcol = self._calculateR2Y(self.Y, self.Yhat_dev)

        if isinstance(cvfold, int) and cvfold > 0:
            self.cvfold = cvfold
            self.Yhatcv = np.zeros((self.n, self.py))
            for i in np.arange(cvfold):
                test = np.arange(i, self.n, cvfold)
                Xtest = self.X[test, :]
                Xtrain = np.delete(self.X, test, axis=0)
                Ytrain = np.delete(self.Y, test, axis=0)

                if self.missingValuesInX:
                    XX = nanmatprod(Xtrain, Xtrain.T)
                else:
                    XX = Xtrain.dot(Xtrain.T)

                # if self.missingValuesInY:
                #     YY = nanmatprod(Ytrain, Ytrain.T)
                # else:
                #     if self.py > 1:
                #         YY = Ytrain.dot(Ytrain.T)
                #     else:
                #         YY = np.outer(Ytrain, Ytrain)

                Tcv, Ucv, Ccv, Kcorrcv, warning = _kpls(XX, Ytrain, auto_penalization=True, ncp=ncp, err_lim=err_lim, nloop_max=nloop_max, warning_tag=False)

                if self.missingValuesInX:
                    Pcv = nanmatprod(Xtrain.T, Tcv.dot(np.linalg.inv(Tcv.T.dot(Tcv))))
                else:
                    Pcv = Xtrain.T.dot(Tcv).dot(np.linalg.inv(Tcv.T.dot(Tcv)))

                # if self.missingValuesInY:
                #     Ccv = nanmatprod(Ytrain.T, Tcv.dot(np.linalg.inv(Tcv.T.dot(Tcv))))
                # else:
                #     Ccv = Ytrain.T.dot(Tcv).dot(np.linalg.inv(Tcv.T.dot(Tcv)))

                Bcv = Pcv.dot(np.linalg.inv(Pcv.T.dot(Pcv))).dot(Ccv.T)
                if Bcv.ndim < 2:
                    Bcv = np.expand_dims(Bcv, axis=1)
                if self.missingValuesInX:
                    self.Yhatcv[test,:] = nanmatprod(Xtest, Bcv)
                else:
                    self.Yhatcv[test,:] = Xtest.dot(Bcv)

            self.Q2Y, self.Q2Ycol = self._calculateR2Y(self.Y, self.Yhatcv)

        else:
            self.Q2Y = "NA"
            self.Q2Ycol = "NA"

        self.R2X = np.sum(np.square(self.T @ self.P.T))/self.SSX
