import pyPLS
import numpy as np
import sys
import time

def test_pls():
    out = pyPLS.nopls(Xt, Yt[:, 0], cvfold=7, scaling=1., penalization=True)
    print("noPLS1 R2 = ", out.R2Y, out.ncp)
    print("noPLS1 Q2 = ", out.Q2Y, out.ncp)
    out = pyPLS.pls(Xt, Yt[:, 0], ncp=2, scaling=1., cvfold=7)
    print("PLS1 R2 = ", out.R2Y, out.ncp)
    print("PLS1 Q2 = ", out.Q2Y, out.ncp)
    out = pyPLS.nopls(Xt, Yt[:, 0:2], cvfold=7, scaling=1., penalization=True)
    print("noPLS2 R2 = ", out.R2Ycol[0], out.R2Ycol[1], out.ncp)
    print("noPLS2 Q2 = ", out.Q2Ycol[0], out.Q2Ycol[1], out.ncp)
    out = pyPLS.pls(Xt, Yt[:, 0:2], ncp=2, cvfold=7, scaling=1.)
    print("PLS2 R2 = ", out.R2Ycol[0], out.R2Ycol[1], out.ncp)
    print("PLS2 Q2 = ", out.Q2Ycol[0], out.Q2Ycol[1], out.ncp)

def test_kernel():
    start_time = time.time()
    K2 = pyPLS.linear(Xt, Y=Xt)
    print("Linear Kernel --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    K2 = pyPLS.gaussian(Xt, sigma=3.0)
    print("Gaussian Kernel --- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    Xt, Z, Yt = pyPLS.simulateData(50, 2, 1000, 5., signalToNoise=100.)
    if "nopls" in sys.argv[1:]:
        test_pls()
    if "kernel" in sys.argv[1:]:
        test_kernel()