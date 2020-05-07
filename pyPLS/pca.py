import numpy as np
import pandas as pd

from pyPLS import preprocessing
from pyPLS._PLSbase import lvmodel
from pyPLS.utilities import isValid
from .engines import pca as _pca, nipals, longtable_pca



class prcomp(lvmodel):
    """
    Principal component analysis
        Parameters:
            X: {N, P} array like
                a table of N observations (rows) and P variables (columns) - The explanatory variables,
            a: The number of component to be fitted
            scaling: float, optional
                A number typically between 0.0 and 1.0 corresponding to the scaling, typical example are
                0.0 corresponds to mean centring
                0.5 corresponds to Pareto scaling
                1.0 corresponds to unit variance scaling

        Returns
        -------
        out : a pca object with ncp components
            Attributes:
                ncp: number of components fitted
                T :  scores table
                P :  loadings table
                E :  residual table
                R2X: Part of variance of X explained by individual components
                R2Xcum: Total variance explained by all the components

            methods:
            scores(n), loadings(n)
                n: int
                    component id
                return the scores of the nth component

    """

    def __init__(self, X, a, scaling=0, center=True, method='svd'):

        lvmodel.__init__(self)
        self.model = "pca"
        self.scaling = scaling

        X, self.n, self.p = isValid(X)

        if np.isnan(X).any():
            self.missingValuesInX = True
            self.SSX = np.nansum(np.square(X))
        else:
            self.SSX = np.sum(np.square(X))

        if type(X) == np.ndarray:
            self.scaling = scaling

            X, self.Xbar, self.Xstd = preprocessing.scaling(X, scaling, center=center)
            if method == 'svd':
                self.T, self.P, self.E, self.R2X = _pca(X, a)
            elif method == 'nipals':
                self.T, self.P, self.E, self.R2X = nipals(X, a)
            elif method == 'longTable':
                self.T, self.P, self.E, self.R2X = longtable_pca(X, a)
            else:
                raise ValueError("Unknown method for PCA")

            if self.R2X is None:
                self.R2X = np.sum(np.square(self.T @ self.P.T)) / self.SSX
                self.cumR2X = np.sum(self.R2X)
            else:
                self.cumR2X = np.sum(self.R2X)

            self.ncp = a

        else:
            raise ValueError("Your table (X) as an unsupported type")


    def predict(self, Xnew, preprocessing=True, statistics=False, **kwargs):
        Xnew, nnew, pxnew = isValid(Xnew, forPrediction=True)
        if preprocessing:
            Xnew = (Xnew - self.Xbar)
            Xnew /= np.power(self.Xstd, self.scaling)

        assert pxnew == self.p, "New observations do not have the same number of variables!!"

        That = Xnew @ self.P

        if statistics:
            Xpred = That @ self.P.T
            Xres = Xnew - Xpred
            Xnew2 = np.square(Xres)

            if np.isnan(Xnew2).any():
                ssrx = np.nansum(Xnew2, axis=0)
            else:
                ssrx = np.sum(Xnew2, axis=0)
            stats = {'That': That, 'ESS': ssrx}
            return That, stats

        else:
            return That

    def summary(self):
        missing_values = np.sum(np.isnan(self.E))
        missing_value_ratio = missing_values / (self.p*self.n)
        print("Summary of input table")
        print("----------------------")
        print("Observations: " + str(self.n))
        print("Variables: " + str(self.p))
        print("Missing values: " + str(missing_values) + " (" + str(missing_value_ratio) + "%)")
        print()
        print("Summary of PCA:")
        print("---------------")
        print("Number of components: " + str(self.ncp))
        print("Total Variance explained: " + str(np.round(self.cumR2X,3)*100)+ "%")
        print("Variance explained by component:")
        for i, r2x in enumerate(self.R2X):
            print("    - Component " + str(i+1) + " : " + str(np.round(r2x,3)*100)+ "%")

if __name__ == '__main__':
    Xt = np.random.randn(20, 100)
    yt = np.random.randn(20, 1)
    Xt[0, 6] = np.nan
    pc = prcomp(Xt, 3, scaling=1)
    pc.summary()


