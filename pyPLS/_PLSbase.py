import numpy as np
from .utilities import isValid, nanmatprod
from .preprocessing import scaling as prepscaling
from .kernel import IMPLEMENTED_KERNEL as kernels


class lvmodel(object):
    def __init__(self):
        self.T = None
        self.P = None

        self.n = 0
        self.ncp = 0
        self.R2X = None
        self.Q2Y = None
        self.model = "LV"
        self.R2Ycol = []
        self.Q2Ycol = []

        self.warning = None

    def scores(self, n):
        if self.T is not None:
            if n == 0:
                # TODO Raise an exception rather than returning None
                return None
            return np.asarray(self.T[:, n-1])
        else:
            return None

    def loadings(self, n):
        if self.P is not None:
            if n >= 1:
                return np.asarray(self.P[:, n-1])
            else:
                return None
        else:
            return None


class plsbase(lvmodel):
    def __init__(self, X, Y, ncp=None, scaling=0, statistics=True, cvfold=7):
        lvmodel.__init__(self)
        self.ncp = ncp
        self.Yhat = None
        self.W = None
        self.B = None
        self.Bk = None
        self.Bcv = None
        self.R2Y = None
        self.R2Ycol = None
        self.cvfold = cvfold
        self.Yhatcv = None
        self.penalization = "NA"
        # Checking if X and Y are valid
        X, nx, px = isValid(X)
        Y, ny, py = isValid(Y)

        if nx != ny :
            raise ValueError("X and Y must have the same number of rows")
        else:
            n = nx

        # Check scaling
        if isinstance(scaling, int):
            scaling = float(scaling)

        if not isinstance(scaling, float):
            raise ValueError("scaling must be a number")

        if scaling >= 0:
            self.X, self.Xbar, self.Xstd = prepscaling(X, scaling)
            self.Y, self.Ybar, self.Ystd = prepscaling(Y, scaling)
        else:
            self.X = X
            self.Y = Y

        self.scaling = scaling

        self.missingValuesInX = False
        self.missingValuesInY = False
        if statistics:
            if np.isnan(X).any():
                self.missingValuesInX = True
                self.SSX = np.nansum(np.square(self.X))
            else:
                self.SSX = np.sum(np.square(self.X))
            if np.isnan(Y).any():
                self.missingValuesInY = True
                self.SSY = np.nansum(np.square(self.Y))
            else:
                self.SSY = np.sum(np.square(self.Y))

            if np.isnan(Y).any():
                self.SSYcol = np.nansum(np.square(self.Y), axis= 0)
            else:
                self.SSYcol = np.sum(np.square(self.Y), axis=0)

        self.n = n
        self.px = px
        self.py = py

    def weights(self, n):
        if self.W is not None:
            return np.array(self.W[:, n-1])
        else:
            return None

    def cross_validation(self,  **kwargs):
        if isinstance(self.cvfold, int) and self.cvfold > 0:
            self.Bcv = np.zeros((self.px,self.py, self.cvfold))
            self.Yhatcv = np.zeros((self.n, self.py))
            for i in np.arange(self.cvfold):
                test = np.arange(i, self.n, self.cvfold)
                Xtest = self.X[test, :]
                Xtrain = np.delete(self.X, test, axis=0)
                Ytrain = np.delete(self.Y, test, axis=0)
                plscv = self.__class__(Xtrain, Ytrain, **kwargs)

                self.Bcv[:, :, i] = plscv.B
                self.Yhatcv[test,:] = plscv.predict(Xtest, preprocessing=False, **kwargs)

            self.Q2Y, self.Q2Ycol = self._calculateR2Y(self.Yhatcv)
        else:
            self.Q2Y = None
            self.Q2Ycol = None

    def predict(self, Xnew, preprocessing=True, **kwargs):

        kernel = None
        for key, value in kwargs.items():
            if key == "kernel":
                kernel = value

        Xnew, nnew, pxnew = isValid(Xnew, forPrediction=True)
        if preprocessing:
            Xnew = (Xnew - self.Xbar)
            Xnew /=  np.power(self.Xstd, self.scaling)

        assert pxnew == self.px, "New observations do not have the same number of variables!!"

        if self.B is not None:
            # Yhat = Xnew @ self.B

            if self.missingValuesInX:
                Yhat = nanmatprod(Xnew, self.B)
            else:
                Yhat = Xnew @ self.B

            if preprocessing:
                Yhat = Yhat * np.power(self.Ystd, self.scaling) + self.Ybar

        elif self.Bk is not None:
            Kt = kernels[kernel](Xnew, Y=self.X)
            Yhat = Kt @ self.Bk
            if preprocessing:
                Yhat = Yhat * (self.Ystd ** self.scaling) + self.Ybar
        else:
            Yhat = None
        return Yhat

    def _calculateR2Y(self, Yhat):
        """
        Method used to calculate R2y and Q2Y
        :param X:
        :param Y (optional):

        :return:
        R2y or Q2y depending of the input
        """

        ssy = np.square(self.Y - Yhat)

        if self.missingValuesInY:
            sserr = np.nansum(ssy)
        else:
            sserr = np.sum(ssy)

        R2Y = 1 - sserr/self.SSY
        if self.py > 1:
            sserr = np.sum(ssy, axis = 0)
            R2Ycol = 1 - sserr/self.SSYcol
        else:
            R2Ycol = None

        return R2Y, R2Ycol

    def summary(self):
        missing_values_inX = np.sum(np.isnan(self.X))
        missing_value_ratio_inX = missing_values_inX / (self.px*self.n)
        missing_values_inY = np.sum(np.isnan(self.Y))
        missing_value_ratio_inY = missing_values_inY / (self.py*self.n)
        print("----------------------")
        print("Summary of input table")
        print("----------------------")
        print("Observations: " + str(self.n))
        print("Predictor Variables (X): " + str(self.px))
        print("Response Variables (Y): " + str(self.py))
        print("Missing values in X: " + str(missing_values_inX) + " (" + str(missing_value_ratio_inX) + "%)")
        print("Missing values in Y: " + str(missing_values_inY) + " (" + str(missing_value_ratio_inY) + "%)")
        print("---------------")
        print("Summary of PLS:")
        print("---------------")
        print("Fitted using " + self.model)
        try:
            print("Penalisation: " + str(self.penalization))
        except:
            pass
        print("Number of components: " + str(self.ncp))
        if self.warning:
            print("Warning: " + self.warning)
        if self.py > 1:
            print("Total explained variance in Y (R2Y): " + str(np.round(self.R2Y,3)))
            print("Determination coefficient by column in Y:")
            for i, r2y in enumerate(self.R2Ycol):
                print("    - Column " + str(i+1) + " : " + str(np.round(r2y,3)))
        else:
            print("Determination coefficient (R2Y): " + str(np.round(self.R2Y,3)))
        if self.R2X:
            print("Modeled variance in X: " + str(np.round(self.R2X,3)))
        if self.Q2Y:
            print("Cross-validation:")
            print("Number of fold: " + str(self.cvfold))
            print("Cumulative Q2Y: " + str(np.round(self.Q2Y,3)))

            if self.py > 1:
                print("Q2 by column in Y:")
                for i, r2y in enumerate(self.Q2Ycol):
                    print("    - Column " + str(i+1) + " : " + str(np.round(r2y,3)))




