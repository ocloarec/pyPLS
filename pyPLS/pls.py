from __future__ import print_function
import numpy as np
import pandas as pd
from _PLSbase import lvmodel
from utilities import nanmatprod
import engines
import preprocessing as prep


class _pls(lvmodel):
    def __init__(self, X, Y, scaling=0):
        lvmodel.__init__(self)

        # Adding weights
        self.W = None
        self.B = None

        # If X is a Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.as_matrix()

        if isinstance(X, list):
            try:
                X = np.asarray(X, dtype=np.float64)
            except ValueError:
                raise ValueError("Cannot cast X into an array of floats")

        if isinstance(X, np.ndarray):

            try:
                n, p = X.shape
            except ValueError:
                raise ValueError("X must be a 2D numpy array")

        else:
            raise ValueError("X must be an array like object")

        if isinstance(Y, pd.DataFrame):
            Y = Y.as_matrix()

        if type(Y) == pd.Series:
            Y = Y.as_matrix()

        if isinstance(Y, list):
            try:
                Y = np.asarray(Y, dtype=np.float64)
            except ValueError:
                raise ValueError("Cannot cast X into an array of floats")

        if isinstance(Y, np.ndarray):
            if Y.shape[0] != n:
                Y = Y.T
                if Y.shape[0] != n:
                    raise ValueError("Y must have a dimension equal to the number of rows in X")

            if Y.ndim > 2:
                raise ValueError("Y cannot have more than 2 dimensions")
        else:
            raise ValueError("Y must be an array like object")

        if Y.ndim < 2:
            Y = np.expand_dims(Y, axis=1)

        n, py = Y.shape

        # Check scaling
        if isinstance(scaling, int):
            scaling = float(scaling)
        if not isinstance(scaling, float):
            raise ValueError("scaling must be a number")

        self.X, self.Xbar, self.Xstd = prep.scaling(X, scaling)
        self.Y, self.Ybar, self.Ystd = prep.scaling(Y, scaling)
        self.scaling = scaling

        self.missingValuesInX = False
        if np.isnan(X).any():
            self.missingValuesInX = True
            self.SSX = np.nansum(self.X**2)
        else:
            self.SSX = np.sum(self.X**2)

        self.missingValuesInY = False
        if np.isnan(Y).any():
            self.missingValuesInY = True
            self.SSY = np.nansum(self.Y**2)
        else:
            self.SSY = np.sum(self.Y**2)

        if np.isnan(Y).any():
            self.SSYcol = np.nansum(self.Y**2, axis= 0)
        else:
            self.SSYcol = np.sum(self.Y**2, axis=0)

        self.n = n
        self.px = p
        self.py = py

    def weights(self, n):
        if self.W is not None:
            return np.array(self.W[:, n-1])
        else:
            return None

    def predict(self, Xnew):
        if self.B is not None:
            Xw = (Xnew - self.Xbar)
            Xw = Xw / (self.Xstd ** self.scaling)

            Yhat = Xw @ self.B

            Yhat = Yhat * (self.Ystd ** self.scaling) + self.Ybar

            return Yhat


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

    """

    def __init__(self, X, y, a, scaling=0, cvfold=7):

        _pls.__init__(self, X, y, scaling=scaling)

        if self.missingValuesInY:
            raise ValueError("noPLS1 does not support missing values in y")

        self.model = "pls1"
        self.T, self.P, self.W, self.C = engines.pls1(self.X, self.Y, a, missing_values=self.missingValuesInX)

        self.Yhat_tc = self.T.dot(self.C)

        self.B = self.W.dot(np.linalg.inv(self.P.T.dot(self.P))).dot(self.C.T)
        if self.missingValuesInX:
            self.Yhat = nanmatprod(self.X, self.B)
            if self.Yhat.ndim < 2:
                self.Yhat = np.expand_dims(self.Yhat, axis=1)
            sserr = np.sum((self.Y - self.Yhat)**2)
        else:
            self.Yhat = self.X.dot(self.B)
            if self.Yhat.ndim < 2:
                self.Yhat = np.expand_dims(self.Yhat, axis=1)
            sserr = np.sum((self.Y - self.Yhat)**2)

        self.R2Y = 1 - sserr/self.SSY

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
                Tcv, Pcv, Wcv, Ccv = engines.pls1(Xtrain, ytrain, a)
                Bcv = Wcv.dot(np.linalg.inv(Pcv.T.dot(Pcv))).dot(Ccv)
                self.Yhatcv[test,0] = Xtest.dot(Bcv)

            sserr = np.sum((self.Y - self.Yhatcv)**2)
            self.Q2Y = 1 - sserr/self.SSY
        else:
            self.Q2Y = "NA"

        self.R2X = np.sum(self.T.dot(self.P.T))/self.SSX


class pls2(_pls):
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

    """
    def __init__(self, X, Y, a, cvfold=None, scaling=0, varMetadata=None, obsMetadata=None):

        _pls.__init__(self, X, Y, scaling=scaling)

        self.model = "pls2"
        missingValues = False
        if self.missingValuesInX or self.missingValuesInY:
            # TODO: For now nissing values in both are dealt the same way Improve this
            missingValues = True
        self.T, self.U, self.P, self.W, self.C = engines.pls2(self.X, self.Y, a, missing_values=missingValues)
        self.B = self.W.dot(np.linalg.inv(self.P.T.dot(self.P))).dot(self.C.T)
        if self.missingValuesInX:
            self.Yhat = nanmatprod(self.X, self.B)
            sserr = np.sum((self.Y - self.Yhat)**2)
        else:
            self.Yhat = self.X.dot(self.B)
            sserr = np.sum((self.Y - self.Yhat)**2, axis=0)

        self.R2Y = 1 - sserr/self.SSY

        if isinstance(cvfold, int) and cvfold > 0:
            self.Yhatcv = np.zeros((self.n, self.py))
            for i in np.arange(cvfold):
                test = np.arange(i, self.n, cvfold)
                Xtest = self.X[test, :]
                Xtrain = np.delete(self.X, test, axis=0)
                ytrain = np.delete(self.Y, test, axis=0)
                Tcv, Ucv, Pcv, Wcv, Ccv = engines.pls2(Xtrain, ytrain, a, missing_values=missingValues)
                Bcv = Wcv.dot(np.linalg.inv(Pcv.T.dot(Pcv))).dot(Ccv.T)
                if missingValues:
                    self.Yhatcv[test, :] = nanmatprod(Xtest, Bcv)
                else:
                    self.Yhatcv[test, :] = Xtest.dot(Bcv)

            sserr = np.sum((self.Y - self.Yhatcv)**2)
            self.Q2Y = 1 - sserr/self.SSY

            if self.SSYcol is not None:
                sserr = np.sum((self.Y - self.Yhatcv)**2, axis = 0)
                self.Q2Ycol = 1 - sserr/self.SSYcol

        else:
            self.Q2Y = "NA"

        self.R2X = np.sum(self.T.dot(self.P.T))/self.SSX


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

        self.model = "mpls1"
        T, C = engines.nopls1(XX, self.Y, ncp=ncp)
        self.ncp = T.shape[1]

        self.C = C
        self.T = T
        if self.missingValuesInX:
            self.P = nanmatprod(self.X.T, self.T.dot(np.linalg.inv(self.T.T.dot(self.T))))
        else:
            self.P = self.X.T.dot(self.T).dot(np.linalg.inv(self.T.T.dot(self.T)))
        self.B = self.P.dot(np.linalg.inv(self.P.T.dot(self.P))).dot(self.C.T)
        if self.missingValuesInX:
            self.Yhat = nanmatprod(self.X, self.B)
            sserr = np.sum((self.Y - self.Yhat)**2)
        else:

            self.Yhat = self.X.dot(self.B)
            sserr = np.sum((self.Y - self.Yhat)**2)

        self.R2Y = 1 - sserr/self.SSY

        if isinstance(cvfold, int) and cvfold > 0:
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
                Tcv, Ccv = engines.nopls1(XXtr, ytrain)
                if self.missingValuesInX:
                    Pcv = nanmatprod(Xtrain.T, Tcv.dot(np.linalg.inv(Tcv.T.dot(Tcv))))
                else:
                    Pcv = Xtrain.T.dot(Tcv).dot(np.linalg.inv(Tcv.T.dot(Tcv)))

                Bcv = Pcv.dot(np.linalg.inv(Pcv.T.dot(Pcv))).dot(Ccv.T)
                self.Yhatcv[test,:] = Xtest.dot(Bcv)

            sserr = np.sum((self.Y-self.Yhatcv)**2)
            self.Q2Y = 1 - sserr/self.SSY
        else:
            self.Q2Y = "NA"

        self.R2X = np.sum(self.T.dot(self.P.T))/self.SSX


class nopls2(_pls):
    """
        This is the multivariate noPLS algorithm.
        Parameters:
            X: {N, P} array like
                a table of N observations (rows) and P variables (columns) - The explanatory variables,
            Y: {N, Q} array like
                a table of N observations (rows) and Q variables (columns) - The dependent variables,
            scaling: float, optional
                A number typically between 0.0 and 1.0 corresponding to the scaling, typical example are
                0.0 corresponds to mean centring
                0.5 corresponds to Pareto scaling
                1.0 corresponds to unit variance scaling
            cvfold: int, optional
                the number of folds in the cross-validation - default is 7

        Returns
        -------
        out : a nopls2 object with ncp components
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

    """
    def __init__(self, X, Y, cvfold=None, scaling=0, varMetadata=None, obsMetadata=None, ncp=None, err_lim=1e-9, nloop_max=200):

        _pls.__init__(self, X, Y, scaling=scaling)

        self.model = "nopls2"
        missingValues = False

        self.err_lim = err_lim
        self.ncp = ncp
        self.nloop_max = nloop_max

        if self.missingValuesInX or self.missingValuesInY:
            # TODO: For now nissing values in both are dealt the same way Improve this
            missingValues = True

        if self.missingValuesInX:
            XX = nanmatprod(self.X, self.X.T)
        else:
            XX = self.X.dot(self.X.T)

        if np.isnan(XX).any():
            raise ValueError("Calculation of XX' gives missing values!")

        if self.missingValuesInY:
            YY = nanmatprod(self.Y, self.Y.T)
        else:
            if self.py > 1:
                YY = self.Y.dot(self.Y.T)
            else:
                YY = np.outer(self.Y, self.Y)

        if np.isnan(YY).any():
            raise ValueError("Calculation of YY' gives missing values!")
        #####################
        self.T, self.U, self.warning = engines.nopls2(XX, YY, ncp=ncp, err_lim=err_lim, nloop_max=nloop_max)
        #####################
        if self.missingValuesInX:
            self.P = nanmatprod(self.X.T, self.T.dot(np.linalg.inv(self.T.T.dot(self.T))))
        else:
            self.P = self.X.T.dot(self.T).dot(np.linalg.inv(self.T.T.dot(self.T)))

        if self.missingValuesInY:
            self.C = nanmatprod(self.Y.T, self.T.dot(np.linalg.inv(self.T.T.dot(self.T))))
        else:
            self.C = self.Y.T.dot(self.T).dot(np.linalg.inv(self.T.T.dot(self.T)))

        self.B = self.P.dot(np.linalg.inv(self.P.T.dot(self.P))).dot(self.C.T)

        if self.missingValuesInX:
            self.Yhat = nanmatprod(self.X, self.B)
            if self.py > 1:
                sserr = np.sum((self.Y - self.Yhat)**2)
            else:
                sserr = np.sum((self.Y - self.Yhat[:,0])**2)
        else:
            self.Yhat = self.X.dot(self.B)
            sserr = np.sum((self.Y - self.Yhat)**2)

        self.R2Y = 1 - sserr/self.SSY

        if isinstance(cvfold, int) and cvfold > 0:
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

                if self.missingValuesInY:
                    YY = nanmatprod(Ytrain, Ytrain.T)
                else:
                    if self.py > 1:
                        YY = Ytrain.dot(Ytrain.T)
                    else:
                        YY = np.outer(Ytrain, Ytrain)

                Tcv, Ucv, warning = engines.nopls2(XX, YY, ncp=ncp, err_lim=err_lim, nloop_max=nloop_max, warning_tag=False)

                if self.missingValuesInX:
                    Pcv = nanmatprod(Xtrain.T, Tcv.dot(np.linalg.inv(Tcv.T.dot(Tcv))))
                else:
                    Pcv = Xtrain.T.dot(Tcv).dot(np.linalg.inv(Tcv.T.dot(Tcv)))

                if self.missingValuesInY:
                    Ccv = nanmatprod(Ytrain.T, Tcv.dot(np.linalg.inv(Tcv.T.dot(Tcv))))
                else:
                    Ccv = Ytrain.T.dot(Tcv).dot(np.linalg.inv(Tcv.T.dot(Tcv)))

                Bcv = Pcv.dot(np.linalg.inv(Pcv.T.dot(Pcv))).dot(Ccv.T)

                if self.missingValuesInX:
                    self.Yhatcv[test,:] = nanmatprod(Xtest, Bcv)
                else:
                    self.Yhatcv[test,:] = Xtest.dot(Bcv)

            sserr = np.sum((self.Y - self.Yhatcv)**2)
            self.Q2Y = 1 - sserr/self.SSY

            if self.SSYcol is not None:
                sserr = np.sum((self.Y - self.Yhatcv)**2, axis = 0)
                self.Q2Ycol = 1 - sserr/self.SSYcol
            else:
                self.Q2Ycol = "NA"

        else:
            self.Q2Y = "NA"
            self.Q2Ycol = "NA"

        self.R2X = np.sum(self.T.dot(self.P.T))/self.SSX


if __name__ == '__main__':
    from simulation import simulateData
    Xt, Z, Yt = simulateData(50, 3, 1000, 50., signalToNoise=100.0)
    print("Simulated data has 3 components")
    print("Testing noPLS1...")
    out = nopls1(Xt, Yt[:, 0], cvfold=7)
    print("Number of components: " + str(out.ncp))
    print("R2Y = " + str(out.R2Y))
    print("Q2Y = " + str(out.Q2Y))
    print("Orthogonality of T : " + str(np.linalg.det(out.T.T.dot(out.T))*100) + "%")
    print("Prediction ")
    print("Real value: " + str(Yt[0,0]))
    print("Predicted value: " + str(out.predict(Xt[0,:])))
    print()

    print("Testing noPLS2 with one column in Y...")
    out = nopls2(Xt, Yt[:, 0], cvfold=7)
    print("Number of components: " + str(out.ncp))
    print("R2Y = " + str(out.R2Y))
    print("Q2Ycol = " + str(out.Q2Ycol))
    print("Q2Y = " + str(out.Q2Y))
    print("Orthogonality of T : " + str(np.linalg.det(out.T.T.dot(out.T))*100) + "%")
    print()

    print("Testing noPLS2 with two columns in Y...")
    out = nopls2(Xt, Yt[:, 0:2], cvfold=7)
    print("Number of components: " + str(out.ncp))
    print("R2Y = " + str(out.R2Y))
    print("Q2Ycol = " + str(out.Q2Ycol))
    print("Q2Y = " + str(out.Q2Y))
    print("Orthogonality of T : " + str(np.linalg.det(out.T.T.dot(out.T))*100) + "%")
    print()

    print("Testing PLS1 with 3 components...")
    out = pls1(Xt, Yt[:, 0], 3, cvfold=7)
    print("R2Y = " + str(out.R2Y))
    print("Q2Y = " + str(out.Q2Y))
    print()
    print("Testing PLS2 with 3 components...")
    out = pls2(Xt, Yt[:, 0:2], 3, cvfold=7)
    print("R2Y = " + str(out.R2Y))
    print("Q2Ycol = " + str(out.Q2Ycol))
    print("Q2Y = " + str(out.Q2Y))
