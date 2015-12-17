import numpy as np
from .utilities import nanmatprod

ELIM = 1e-12


def foo(r, X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    XX = X @ X.T
    n, p = XX.shape
    H = (np.eye(n) - np.ones((n, n), dtype=float)/n)
    diagXX = np.mean(np.diag(XX))
    XX -= diagXX * r * np.identity(n)
    XX = H.T @ XX @ H
    Yproj = XX @ Y
    return np.sum(Yproj * Y, 0)


def foo2(r, X, Y):
    return np.abs(np.mean(foo(r, X, Y)))


def pca(X, a):
    if isinstance(X, np.ndarray):
        # try:
        if np.isnan(X).any():
            XX = nanmatprod(X, X.T)
        else:
            XX = X @ X.T

        u, s, v = np.linalg.svd(XX)
        S = np.eye(len(s))*s
        P = np.zeros((X.shape[1], a))
        T = np.sqrt(S).dot(u.T).T
        T = T[:, :a]
        E = X

        if np.isnan(X).any():
            for i in np.arange(a):
                P[:, i] = nanmatprod(X.T, T[:, i])[:, 0]
                P[:, i] = P[:, i] / np.sqrt(np.sum(P[:, i]**2))
                E = X - np.outer(T[:, i], P[:, i])
        else:
            for i in np.arange(a):
                P[:, i] = X.T @ T[:, i]
                P[:, i] = P[:, i] / np.sqrt(np.sum(P[:, i]**2))
                E = X - np.outer(T[:, i], P[:, i])


        return T, P, E, np.diag(S)[:a]/ np.sum(np.diag(S))

        # except ValueError:
        #     raise ValueError("X must be a 2D numpy array")
    else:
        raise ValueError("X must be a 2D numpy array")


def nipals(X, a):

    P = None
    T = None
    p = np.zeros((X.shape[1],))

    for i in np.arange(a):
        t = np.mean(X, axis=1)
        nloop = 0
        err = np.Inf
        while err > ELIM and nloop < 1000:
            p_old = p
            p = X.T @ t / np.sum(t * t)
            p = p / np.sqrt(sum(p * p))
            t = X @ p
            err = np.sum((p_old - p) * (p_old - p))
            nloop += 1

        if P is None:
            P = p
        else:
            P = np.column_stack((P, p))

        if T is None:
            T = t
        else:
            T = np.column_stack((T, t))

        X -= np.outer(t, p)

    return T, P, X


def pls1(X, y, a, missing_values=False):
    n, p = X.shape
    c = np.zeros(a)
    T = np.zeros((n, a))
    P = np.zeros((p, a))
    W = np.zeros((p, a))

    for i in np.arange(a):
        if missing_values:
            w = nanmatprod(X.T, y)
            w = w / np.sqrt(w.T * w)
            t = nanmatprod(X, w)
            p = nanmatprod(X.T, t) / np.sum(t * t)
        else:
            w = X.T @ y
            w = w / np.sqrt(np.sum(w * w))
            t = X @ w
            p = X.T @ t / np.sum(t * t)

        X = X - np.outer(t, p)
        c[i] = y.T @ t / np.sum(t * t)
        y = y - c[i]*t
        T[:, i] = t[:,0]
        P[:, i] = p[:,0]
        W[:, i] = w[:,0]

    return T, P, W, c


def pls2(X, Y, a, missing_values=False, err_lim=0.00001):
    n, px = X.shape
    try:
        n, py = Y.shape
    except ValueError:
        Y = np.expand_dims(Y, axis=1)
        n, py = Y.shape

    T = np.zeros((n, a))
    U = np.zeros((n, a))
    P = np.zeros((px, a))
    W = np.zeros((px, a))
    C = np.zeros((py, a))

    for i in np.arange(a):
        error = 1
        wo=np.zeros((px, 1))
        if missing_values:
            u = np.nanmean(Y, 1)
            while error > err_lim:
                w = nanmatprod(X.T, u)[:,0]
                w = w/np.sqrt(np.inner(w, w))
                t = nanmatprod(X, w)[:,0]
                c = nanmatprod(Y.T, t) / np.inner(t, t)
                c = c[:, 0]
                u = nanmatprod(Y, c) / np.inner(c, c)
                u = u[:, 0]
                error = np.sum((w - wo)**2)
                wo = w

            p = nanmatprod(X.T, t)[:,0] / np.inner(t, t)

        else:
            u = np.mean(Y, 1)
            while error > err_lim:
                w = X.T @ u
                w = w/np.sqrt(np.inner(w, w))
                t = X @ w
                c = Y.T @ t / np.inner(t, t)
                u = Y @ c / np.inner(c, c)
                error = np.sum((w - wo)**2)
                wo = w

            p = X.T @ t / np.inner(t, t)

        X = X - np.outer(t, p)
        Y = Y - np.outer(t, c)

        T[:, i] = t
        U[:, i] = u
        P[:, i] = p
        W[:, i] = w
        C[:, i] = c

    return T, U, P, W, C


def diagonal_correction(ZZ, v, n):
    correction = np.zeros((n, n))
    H = (np.eye(n) - np.ones((n, n), dtype=float)/n)
    for i, zz in enumerate(ZZ):
        zzw = np.delete(zz, i)
        zzwc = zzw - np.mean(zzw)
        vw = np.delete(v, i)
        a = np.inner(zzwc, vw)/np.sum(vw**2)
        b = np.mean(zzw) - a * np.mean(vw)
        zzihat = v[i]*a + b
        correction[i, i] = zz[i] - zzihat

    ZZ = ZZ - correction
    # ZZ is then recentred
    ZZ = H.T @ ZZ @ H
    return ZZ


def nopls1(XX, y, ncp=None):

    n = XX.shape[0]
    # Array initialisation
    XX = diagonal_correction(XX, y, n)
    if not ncp:
        ncp = np.linalg.matrix_rank(XX)
        yXXylim = 0
    else:
        yXXylim = -np.inf

    T = np.zeros((n, ncp))
    C = np.zeros((1, ncp))

    nc = 0
    yXXy = 1.

    while yXXy > yXXylim and nc < ncp:
        nc += 1
        t = XX @ y
        t = t/np.sqrt(np.sum(t**2))
        c = y.T @ t
        pp = t.T @ XX @ t
        Xpt = XX @ np.outer(t, t)
        tpX = np.outer(t, t) @ XX
        # Deflation of y
        y = y - c * t
        # Deflation of XX
        XX = XX - Xpt - tpX + pp * np.outer(t, t)
        try:
            T[:, nc-1] = t[:, 0]
            C[0, nc-1] = c[0]
        except:
            T[:, nc-1] = t
            C[0 ,nc-1] = c
        yXXy = y.T @ XX @ y

    T = np.delete(T, np.s_[nc:], 1)
    C = np.delete(C, np.s_[nc:], 1)

    return T, C


def nopls2(XX, YY, ncp=None, err_lim=1e-9, nloop_max=200, warning_tag=True, penalizeY=True):
    n = XX.shape[0]
    # Array initialisation
    if ncp is None:
        ncp = np.linalg.matrix_rank(XX)
    T = np.zeros((n,ncp))
    U = np.zeros((n,ncp))
    # Not true for n < py but it is used to know if we have single or multi Y
    py = np.linalg.matrix_rank(YY)

    nc = 0
    uXXu = 1.

    u = YY[:,0]

    if not ncp:
        ncp = np.inf
        uXXulim = 0
    else:
        uXXulim = -np.inf

    nloop = 0
    warning = None
    while nc < ncp:
        nc += 1
        err = np.inf
        while err > err_lim and nloop < nloop_max:
            # Applying the correction on the diagonal of XX
            if nc < 2:
                XX = diagonal_correction(XX, u, n)

            t = XX @ u
            t = t / np.sqrt(np.sum(t**2))

            # Applying the correction on the diagonal of YY
            if nc < 2 and penalizeY:
                YY = diagonal_correction(YY, t, n)
            u_new = YY @ t
            u_new = u_new/np.sqrt(np.sum(u_new**2))

            err = np.sum((u - u_new)**2)
            u = u_new
            nloop += 1

        if nloop < nloop_max or nc == 1:
            # Deflation of XX
            pp = t.T @ XX @ t / np.inner(t, t)
            Xpt = XX @ np.outer(t, t) / np.inner(t, t)
            tpX = np.outer(t, t) @ XX / np.inner(t, t)
            XX = XX - Xpt - tpX + pp * np.outer(t, t)
            # Deflation of YY
            cc = t.T @ YY @ t / np.inner(t, t)
            Yct = YY @ np.outer(t, t) / np.inner(t, t)
            tcY = np.outer(t, t) @ YY / np.inner(t, t)
            YY = YY - Yct - tcY + cc * np.outer(t, t)

            try:
                T[:, nc-1] = t[:, 0]
                U[:, nc-1] = u[:, 0]
            except:
                T[:, nc-1] = t
                U[:, nc-1] = u

            if nloop >= nloop_max:
                T = np.delete(T, np.s_[nc:], 1)
                U = np.delete(U, np.s_[nc:], 1)
                if warning_tag:
                    warning = "Component " + str(nc) + " has not converged."
                break
        else:
            T = np.delete(T, np.s_[nc-1:], 1)
            U = np.delete(U, np.s_[nc-1:], 1)
            break

    return T, U, warning




