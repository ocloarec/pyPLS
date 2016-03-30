from __future__ import print_function
import numpy as np
from ._PLSbase import plsbase as pls_base
from .utilities import nanmatprod
from .engines import pls as pls_engine


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
        pls_base.__init__(self, X, Y, ncp=ncp, scaling=scaling, cvfold=cvfold)
        self.model = "pls"
        missingValues = False
        if self.missingValuesInX or self.missingValuesInY:
            # TODO: For now nissing values in both X and Y are dealt the same way -> Improve this
            missingValues = True
        self.T, self.U, self.P, self.W, self.C, self.B = pls_engine(self.X, self.Y, self.ncp, missing_values=missingValues)
        self.Yhat = self.predict(self.X, preprocessing=False)
        self.R2Y, self.R2Ycol = self._calculateR2Y(self.Yhat)
        self.cross_validation(ncp=ncp)
        self.R2X = np.sum(np.square(self.T @ self.P.T))/self.SSX

