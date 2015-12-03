import numpy as np
from scipy.stats import t, norm


class preliminaryExperiment(object):
    """
    An tool to estimate the optimum number of samples when designing experiments
    """
    def __init__(self, x1, x2):
        self.m1 = np.mean(x1)
        self.s1 = np.std(x1)
        self.m2 = np.mean(x2)
        self.s2 = np.std(x2)
        self.n1 = len(x1)
        self.n2 = len(x2)
        self.pooled_s = np.sqrt(((self.n1 - 1) * self.s1**2 + (self.n2 - 1) * self.s2**2) / (self.n1 + self.n2 - 2))


    def sample_size(self, alpha, beta, nlimit=10000):
        """
        :param alpha: risk alpha
        :param beta: risk beta
        :param nlimit:
        :return: an estimate of the required sample size
        """
        n = 3
        d = np.abs(self.m1 - self.m2)
        tbeta = t.isf(beta, n)
        talpha = t.isf(alpha, self.n1)
        while d*np.sqrt(n)/self.pooled_s < (tbeta+talpha):

            n += 1
            tbeta = t.isf(beta, n)
            if n > nlimit:
                break
        return n

    def predictionConfidenceInterval(self, alpha):
        mu = np.abs(self.m1 - self.m2)
        z = norm.isf(alpha/2)
        mulow = mu - z*self.s2/np.sqrt(self.n2)
        muhigh = mu + z*self.s2/np.sqrt(self.n2)
        return norm.cdf(mulow/(2*self.pooled_s)), norm.cdf(muhigh/(2*self.pooled_s))



if __name__ == '__main__':
    x1 = np.random.normal(loc=1.0, size=200)
    x2 = np.random.normal(loc=5.0, size=10)
    se = preliminaryExperiment(x1, x2)
    print(se.sample_size(0.05, 0.05))
    print(se.predictionConfidenceInterval(0.05))






