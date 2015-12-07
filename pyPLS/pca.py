import numpy as np
import pandas as pd

from pyPLS import preprocessing
from pyPLS._PLSbase import lvmodel
import engines


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

    def __init__(self, X, a, scaling=0, varMetadata=None):

        lvmodel.__init__(self)
        self.model = "pca"

        if type(X) == pd.DataFrame:
            if not varMetadata:
                self.varMetadata = X.columns
            else:
                # TODO: check varMetadata consistency
                self.varMetadata = varMetadata
            X = X.as_matrix()

        if type(X) == np.ndarray:
            X, self.Xbar, self.Xstd = preprocessing.scaling(X, scaling)
            self.T, self.P, self.E, self.R2X = engines.pca(X, a)
            self.cumR2X = np.sum(self.R2X)
        else:
            raise ValueError("Your table (X) as an unsupported type")
    def summary(self):
        # TODO: Implement a summary
        pass

if __name__ == '__main__':
    Xt = np.random.randn(20, 100)
    yt = np.random.randn(20, 1)
    # Xt[0, 6] = np.nan
    pc = prcomp(Xt, 3, scaling=1)
    print(pc.R2X)


