import numpy as np
from .utilities import isValid
from .preprocessing import scaling as prepscaling

class lvmodel(object):
    def __init__(self):
        self.T = None
        self.P = None

        self.n = 0
        self.ncp = 0
        self.R2X = None
        self.model = "LV"


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
            return np.asarray(self.P[:, n-1])
        else:
            return None


class _pls(lvmodel):
    def __init__(self, X, Y, scaling=0):
        lvmodel.__init__(self)

        # Adding weights
        self.W = None
        self.B = None
        self.R2Y = None
        self.R2Ycol = None
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

        self.X, self.Xbar, self.Xstd = prepscaling(X, scaling)
        self.Y, self.Ybar, self.Ystd = prepscaling(Y, scaling)
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
        self.px = px
        self.py = py

    def weights(self, n):
        if self.W is not None:
            return np.array(self.W[:, n-1])
        else:
            return None

    def predict(self, Xnew):
        if self.B is not None:
            Xnew, nnew, pxnew = isValid(Xnew, forPrediction=True)
            if pxnew == self.px:
                Xw = (Xnew - self.Xbar)
                Xw = Xw / (self.Xstd ** self.scaling)

                Yhat = Xw @ self.B

                Yhat = Yhat * (self.Ystd ** self.scaling) + self.Ybar

                return Yhat

            else:
                raise ValueError("New observations do not have the same number of variables!!")

    def _calculateR2Y(self, realY, predictedY):
        """
        Static method used to calculate R2y and Q2Y
        :param realY:
        :param predictedY:
        :param missingValuesInY:
        :return:
        """
        ssy = (realY - predictedY)**2

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
        print("Summary of input table")
        print("----------------------")
        print("Observations: " + str(self.n))
        print("Predictor Variables (X): " + str(self.px))
        print("Response Variables (Y): " + str(self.py))
        print("Missing values in X: " + str(missing_values_inX) + " (" + str(missing_value_ratio_inX) + "%)")
        print("Missing values in Y: " + str(missing_values_inY) + " (" + str(missing_value_ratio_inY) + "%)")
        print()
        print("Summary of PLS:")
        print("---------------")
        print("Fitted using " + self.model)
        print("Number of components: " + str(self.ncp))
        if self.py > 1:
            print("Total explained variance in Y (R2Y): " + str(np.round(self.R2Y,3)))
            print("Determination coefficient by column in Y:")
            for i, r2y in enumerate(self.R2Ycol):
                print("    - Column " + str(i+1) + " : " + str(np.round(r2y,3)))
        else:
            print("Determination coefficient (R2Y): " + str(np.round(self.R2Y,3)))
        print("Modeled variance in X: " + str(np.round(self.R2X,3)))
        # for i, r2x in enumerate(self.R2X):
        #     print("    - Component " + str(i+1) + " : " + str(np.round(r2x,3)*100)+ "%")





