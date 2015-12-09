import numpy as np
import pandas as pd

from pyPLS import preprocessing
from pyPLS._PLSbase import lvmodel
from pyPLS.utilities import isValid
from .engines import pca as _pca


class prcomp(lvmodel):
    """
    Principal component analysis
        Parameters:
            X: {N, P} array like
                a table of N observations (rows) and P variables (columns) - The explanatory variables,
            scaling: float, optional
                A number typically between 0.0 and 1.0 corresponding to the scaling, typical example are
                0.0 corresponds to mean centring
                0.5 corresponds to Pareto scaling
                1.0 corresponds to unit variance scaling

        Returns
        -------
        out : a nopls2 object with ncp components
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

    def __init__(self, X, a, scaling=0):

        lvmodel.__init__(self)
        self.model = "pca"

        X, self.n, self.p = isValid(X)

        if type(X) == np.ndarray:
            X, self.Xbar, self.Xstd = preprocessing.scaling(X, scaling)
            self.T, self.P, self.E, self.R2X = _pca(X, a)
            self.ncp = a
            self.cumR2X = np.sum(self.R2X)
        else:
            raise ValueError("Your table (X) as an unsupported type")

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


