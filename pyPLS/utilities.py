import numpy as np

def nanmatprod(ma, mb):
    """
    Compute matrix product of a and b in cas there is missing values.
    :type ma: a numpy array or matrix
    :type mb: a numpy array or matrix
    :return: a numpy array or matrix
    """

    ma_was_matrix = False
    if type(ma) == np.matrix:
        ma = np.array(ma)
        ma_was_matrix = True

    mb_was_matrix = False
    if type(mb) == np.matrix:
        mb = np.array(mb)
        mb_was_matrix = True

    nra = ma.shape[0]
    try:
        nca = ma.shape[1]
    except IndexError:
        nca = 1

    nrb = mb.shape[0]
    try:
        ncb = mb.shape[1]
    except IndexError:
        ncb = 1
    mc = np.zeros((nra, ncb))

    if nca == nrb:
        if nra > 1:
            if ncb > 1:
                for i in np.arange(nra):
                    mc[i, :] = np.nansum(mb.T * ma[i, :], 1)
            else:
                for i in np.arange(nra):
                    mc[i, :] = np.nansum(mb.T * ma[i, :])
        else:
            mc[0, :] = np.nansum(mb.T * ma, 1)
    else:
        raise ValueError("shapes " + str(ma.shape) + " and " + str(mb.shape) + " are not aligned")

    if ma_was_matrix and mb_was_matrix:
        mc = np.matrix(mc)

    return mc


def ROCanalysis(Yhat, classes, positive, npoints=50):
    if isinstance(positive, str) or isinstance(positive, unicode):
            positive = [positive]
    classes = [di for di in classes]
    minLim = min(Yhat)
    maxLim = max(Yhat)
    limit = np.arange(minLim, maxLim, (maxLim-minLim)/npoints)
    falsePositive = np.zeros(len(limit))
    falseNegative = np.zeros(len(limit))
    truePositive = np.zeros(len(limit))
    trueNegative = np.zeros(len(limit))

    for k, lim in enumerate(limit):
        for i, y in enumerate(Yhat):
            if y < lim:
                if classes[i] not in positive:
                    falseNegative[k] += 1
                else:
                    trueNegative[k] += 1

            else:
                if classes[i] in positive:
                    falsePositive[k] += 1
                else:
                    truePositive[k] += 1

    sensitivity = truePositive / (truePositive + falseNegative)
    specificity = trueNegative / (trueNegative + falsePositive)

    return specificity, sensitivity


if __name__ == '__main__':
    a = np.random.randn(30, 20)
    a[0,6] = np.nan
    b = np.random.randn(20,6)
    print(nanmatprod(a, b))
