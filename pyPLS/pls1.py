from __future__ import print_function
import numpy as np
from ._PLSbase import _pls
from .utilities import nanmatprod
from .engines import pls1 as _pls1


class pls1(_pls):
    """
    This is the classic univariate NIPALS PLS algorithm.
    Parameters
    ----------
        X: {N, P} array like
            a table of N observations (rows) and P variables (columns) - The explanatory variables,
        y: {N, 1} array like
            a list of N the corresponding values of the dependent variable,
        a: int
            the number of PLS component to be fitted
        scaling: float, optional
            A number typically between 0.0 and 1.0 corresponding to the scaling, typical example are
            0.0 corresponds to mean centring
            0.5 corresponds to Pareto scaling
            1.0 corresponds to unit variance scaling
        cvfold: int, optional
            the number of folds in the cross-validation - default is 7

    Returns
    -------
    out : a pls1 object with a components
        Attributes:
            W : PLS weights table
            T : PLS scores table
            P : PLS loadings table
            C : PLS score regression coefficients
            B : PLS regression coefficients
            Yhat: model predicted Y
            Yhatcv: cross-validation predicted Y
            R2Y: Determination coefficient of Y
            Q2Y: Cross validation parameter

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

    def __init__(self, X, y, a, scaling=0, cvfold=7):

        _pls.__init__(self, X, y, scaling=scaling)

        if self.missingValuesInY:
            raise ValueError("noPLS1 does not support missing values in y")

        self.model = "pls1"
        self.ncp = a
        self.T, self.P, self.W, self.C = _pls1(self.X, self.Y, a, missing_values=self.missingValuesInX)

        self.Yhat_tc = self.T.dot(self.C)
        # Regression coefficient and model prediction
        self.B = self.W.dot(np.linalg.inv(self.P.T.dot(self.P))).dot(self.C.T)
        if self.missingValuesInX:
            self.Yhat = nanmatprod(self.X, self.B)
            if self.Yhat.ndim < 2:
                self.Yhat = np.expand_dims(self.Yhat, axis=1)
        else:
            self.Yhat = self.X.dot(self.B)
            if self.Yhat.ndim < 2:
                self.Yhat = np.expand_dims(self.Yhat, axis=1)

        self.R2Y, self.R2Ycol = self._calculateR2Y(self.Y, self.Yhat)

        if isinstance(cvfold, int) and cvfold > 0:
            self.Yhatcv = np.zeros((self.n,1))
            for i in np.arange(cvfold):
                test = np.arange(i, self.n, cvfold)
                Xtest = self.X[test, :]
                Xtrain = np.delete(self.X, test, axis=0)
                ytrain = np.delete(self.Y, test)
                # TODO: Easy fix to improve
                if ytrain.ndim < 2:
                    ytrain = np.expand_dims(ytrain, axis=1)
                Tcv, Pcv, Wcv, Ccv = _pls1(Xtrain, ytrain, a)
                Bcv = Wcv.dot(np.linalg.inv(Pcv.T.dot(Pcv))).dot(Ccv)
                self.Yhatcv[test,0] = Xtest.dot(Bcv)

            self.Q2Y, self.Q2Ycol = self._calculateR2Y(self.Y, self.Yhatcv)

        else:
            self.Q2Y = "NA"

        self.R2X = np.sum((self.T @ self.P.T)**2)/self.SSX
