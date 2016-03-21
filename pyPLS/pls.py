from __future__ import print_function
import numpy as np
from ._PLSbase import plsbase as pls_base
from .utilities import nanmatprod
from .engines import pls as _pls


class pls(pls_base):
    """
    This is the classic multivariate NIPALS PLS algorithm.
    Parameters:
        X: {N, P} array like
            a table of N observations (rows) and P variables (columns) - The explanatory variables,
        Y: {N, Q} array like
            a table of N observations (rows) and Q variables (columns) - The dependent variables,
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
        out : a pls2 object with a components
            Attributes:
                W : PLS weights table
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
    def __init__(self, X, Y, ncp=1, cvfold=None, scaling=0):

        pls_base.__init__(self, X, Y, scaling=scaling)

        self.model = "pls"
        self.ncp = ncp
        missingValues = False
        if self.missingValuesInX or self.missingValuesInY:
            # TODO: For now nissing values in both are dealt the same way Improve this
            missingValues = True
        self.T, self.U, self.P, self.W, self.C = _pls(self.X, self.Y, self.ncp, missing_values=missingValues)

        # Regression coefficient and model prediction
        self.B = self.W.dot(np.linalg.inv(self.P.T.dot(self.P))).dot(self.C.T)
        if self.missingValuesInX:
            self.Yhat = nanmatprod(self.X, self.B)
        else:
            self.Yhat = self.X.dot(self.B)

        self.R2Y, self.R2Ycol = self._calculateR2Y(self.Yhat)

        self.cvfold = cvfold
        self.cross_validation()

        self.R2X = np.sum((self.T @ self.P.T)**2)/self.SSX

