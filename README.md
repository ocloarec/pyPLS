# pyPLS

PyPLS is a package built around PLS and especially the implementation of noPLS (no overfitting Partial Least Squares)

It also provides a few tools to facilitate the use of and test PLS and noPLS:

- Normalisation
- Preprocessing
- Simulation
- PCA

# Prerequisites
pPLS requires the following packages to be installed on a python > 3.6 distribution: numpy pandas scipy

`pip install numpy pandas scipy`

To plot the output of the different function `matplotlib` could be also installed. It is not a requirement but it is always useful to have it installed aside.

`pip install matplotlib`

# Installation

As it is still a package under development, it is advised to install it as such. This way code can me modified and tested directly in the same notebook environment. This also means that after installation, a simple pull from the GitHub repository could provide you. First clone the repo with:

`git clone https://github.com/ocloarec/pyPLS.git`

then 

`cd pyPLS`
`python setup.py install develop`

That's it, you can now start to use this package. 

