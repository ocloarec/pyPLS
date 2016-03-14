import pyPLS
import numpy as np

if __name__ == '__main__':
    Xt, Z, Yt = pyPLS.simulateData(50, 5, 1000, 10., signalToNoise=100.)

    print("Testing noPLS with one column in Y...")
    out = pyPLS.nopls(Xt, Yt[:, 0], cvfold=7)
    out.summary()
    print()
    print()

    print("Testing noPLS with two columns in Y...")
    out = pyPLS.nopls(Xt, Yt[:, 0:2], cvfold=7)
    out.summary()

    print("Testing noPLS-DA...")
    Ybar = np.mean(Yt[:,0])
    Group1 = np.where(Yt[:,0] < Ybar)
    Group2 = np.where(Yt[:,0] >=Ybar)
    Dummy = np.zeros((50,1))
    Dummy[Group2,:] = 1
    out = pyPLS.nopls(Xt,Dummy, scaling=1., cvfold=7)
    out.summary()