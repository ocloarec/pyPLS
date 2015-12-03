import numpy as np

class lvmodel(object):
    def __init__(self):
        self.T = None
        self.P = None
        self.W = None
        self.n = 0

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





