import pyPLS
import numpy as np
import sys
import time

def test_nopls():
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
    Group2 = np.where(Yt[:,0] >=Ybar)
    Dummy = np.zeros((50,1))
    Dummy[Group2,:] = 1
    out = pyPLS.nopls(Xt,Dummy, scaling=1., cvfold=7)
    out.summary()

    out = pyPLS.nopls(Xt,Dummy, scaling=1., cvfold=7, kernel="gaussian", sigma=100)
    out.summary()

def test_kernel():
    start_time = time.time()
    K2 = pyPLS.linear(Xt, Y=Xt)
    print("Linear Kernel --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    K2 = pyPLS.gaussian(Xt, sigma=3.0)
    print("Gaussian Kernel --- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    Xt, Z, Yt = pyPLS.simulateData(50, 5, 1000, 10., signalToNoise=100.)
    if "nopls" in sys.argv[1:]:
        test_nopls()
    if "kernel" in sys.argv[1:]:
        test_kernel()