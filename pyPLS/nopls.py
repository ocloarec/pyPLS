from __future__ import print_function
import numpy as np
from ._PLSbase import plsbase
from .utilities import nanmatprod
from .preprocessing import diagonal_correction
from .engines import kpls
from .kernel import IMPLEMENTED_KERNEL as kernels

class nopls(plsbase):
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
    def __init__(self, X, Y, scaling=0,
                 kernel="linear",
                 penalization=True,
                 cvfold=None,
                 ncp=None,
                 err_lim=1e-9,
                 nloop_max=200,
                 statistics=True,
                 **kwargs):

        plsbase.__init__(self, X, Y, scaling=scaling, statistics=statistics, cvfold=cvfold)

        self.model = "nopls"
        self.err_lim = err_lim
        self.nloop_max = nloop_max
        self.penalization = penalization

        assert kernels[kernel], "Kernel not supported!"

        self.K = kernels[kernel](self.X, **kwargs)
         # Correction of the kernel matrix
        if penalization:

            self.K = diagonal_correction(self.K, np.mean(self.Y, axis=1))

        assert not np.isnan(self.K).any(), "Kernel calculation lead to missing values!"

        if self.missingValuesInY:
            YY = nanmatprod(self.Y, self.Y.T)
        else:
            YY = self.Y @ self.Y.T

        assert not np.isnan(YY).any(), "A row of Y contains only missing values!"

        #####################
        self.T, self.U, self.C, self.warning = kpls(self.K, self.Y, ncp=ncp, err_lim=err_lim, nloop_max=nloop_max)
        #####################

        # Deduction of the number of component fitted from the score array
        self.ncp = self.T.shape[1]

        if kernel == "linear":
            if self.missingValuesInX:
                self.P = nanmatprod(self.X.T, self.T.dot(np.linalg.inv(self.T.T.dot(self.T))))
            else:
                self.P = self.X.T.dot(self.T).dot(np.linalg.inv(self.T.T.dot(self.T)))

            # Regression coefficient and model prediction
            # self.B = self.X.T @ self.U @ np.linalg.inv(self.T.T @ self.Kcorr @ self.U) @ self.T.T @ self.Y
            self.B = self.P @ np.linalg.inv(self.P.T @ self.P).dot(self.C.T)
            if self.B.ndim < 2:
                self.B = np.expand_dims(self.B, axis=1)
        else:
            self.P = None
            self.B = None
            self.Bk = self.U @ np.linalg.inv(self.T.T @ self.K @ self.U) @ self.T.T @self.Y

        self.cross_validation(scaling=-1,
                              kernel=kernel,
                              penalization=penalization,
                              err_lim=1e-9,
                              nloop_max=200,
                              statistics=True,
                              **kwargs)

        if statistics:
            #self.Yhat = self.predict(self.X, preprocessing=False, kernel=kernel)
            self.Yhat = self.T @ self.C.T
            self.R2Y, self.R2Ycol = self._calculateR2Y(self.Yhat)
            if kernel == "linear":
                self.R2X = np.sum(np.square(self.T @ self.P.T))/self.SSX
            else:
                self.R2X = None

