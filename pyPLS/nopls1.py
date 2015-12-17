from __future__ import print_function
import numpy as np
from ._PLSbase import _pls
from .utilities import nanmatprod
from .engines import nopls1 as _nopls1


class nopls1(_pls):
    """
        This is the univariate noPLS algorithm.
        Parameters:
            X: {N, P} array like
                a table of N observations (rows) and P variables (columns) - The explanatory variables,
            Y: {N, 1} array like
                a list of N the corresponding values of the dependent variable,
            ncp: int
                a fixed number of component to be fitted. Default the number of component is found by the algorithm itself. See algorithm description
            scaling: float, optional
                A number typically between 0.0 and 1.0 corresponding to the scaling, typical example are
                0.0 corresponds to mean centring
                0.5 corresponds to Pareto scaling
                1.0 corresponds to unit variance scaling
            cvfold: int, optional
                the number of folds in the cross-validation - default is 7

        Returns
        -------
        out : a nopls1 object with ncp components
            Attributes:
                ncp: number of components fitted
                T : PLS scores table
                P : PLS loadings table
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


    def __init__(self, X, Y,
                 ncp=None,
                 scaling=0,
                 cvfold=None,
                 varMetadata=None,
                 obsMetadata=None):

        _pls.__init__(self, X, Y, scaling=scaling)

        if self.missingValuesInY:
            raise ValueError("noPLS1 does not support missing values in y")

        if self.missingValuesInX:
            XX = nanmatprod(self.X, self.X.T)
        else:
            XX = self.X.dot(self.X.T)

        if np.isnan(XX).any():
            raise ValueError("Calculation of XX' leads to missing values!")

        self.model = "nopls1"
        T, C = _nopls1(XX, self.Y, ncp=ncp)
        self.ncp = T.shape[1]

        self.C = C
        self.T = T
        if self.missingValuesInX:
            self.P = nanmatprod(self.X.T, self.T.dot(np.linalg.inv(self.T.T.dot(self.T))))
        else:
            self.P = self.X.T.dot(self.T).dot(np.linalg.inv(self.T.T.dot(self.T)))

        # Regression coefficient and model prediction
        self.B = self.P.dot(np.linalg.inv(self.P.T.dot(self.P))).dot(self.C.T)
        if self.missingValuesInX:
            self.Yhat = nanmatprod(self.X, self.B)
        else:
            self.Yhat = self.X.dot(self.B)

        self.R2Y, self.R2Ycol = self._calculateR2Y(self.Y, self.Yhat)

        if isinstance(cvfold, int) and cvfold > 0:
            self.cvfold = cvfold
            self.Yhatcv = np.zeros((self.n, 1))
            for i in np.arange(cvfold):
                test = np.arange(i, self.n, cvfold)
                Xtest = self.X[test, :]
                Xtrain = np.delete(self.X, test, axis=0)
                if self.missingValuesInX:
                    XXtr = nanmatprod(Xtrain, Xtrain.T)
                else:
                    XXtr = Xtrain.dot(Xtrain.T)
                ytrain = np.delete(self.Y, test)
                Tcv, Ccv = _nopls1(XXtr, ytrain)
                if self.missingValuesInX:
                    Pcv = nanmatprod(Xtrain.T, Tcv.dot(np.linalg.inv(Tcv.T.dot(Tcv))))
                else:
                    Pcv = Xtrain.T.dot(Tcv).dot(np.linalg.inv(Tcv.T.dot(Tcv)))

                Bcv = Pcv.dot(np.linalg.inv(Pcv.T.dot(Pcv))).dot(Ccv.T)
                self.Yhatcv[test,:] = Xtest.dot(Bcv)

            self.Q2Y, self.Q2Ycol = self._calculateR2Y(self.Y, self.Yhatcv)
        else:
            self.Q2Y = "NA"

        self.R2X = np.sum((self.T @ self.P.T)**2)/self.SSX
