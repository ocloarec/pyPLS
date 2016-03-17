import numpy as np
from .utilities import nanmatprod

ELIM = 1e-12


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


def pls(X, Y, a, missing_values=False, err_lim=0.00001):
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


def diagonal_correction(K, v):
    n = K.shape[0]
    correction = np.zeros((n, n))
    H = (np.eye(n) - np.ones((n, n), dtype=float)/n)
    for i, k in enumerate(K):
        kw = np.delete(k, i)
        kwc = kw - np.mean(kw)
        vw = np.delete(v, i)
        vwc = vw - np.mean(vw)
        a = np.inner(kwc, vw)/np.sum(vwc**2)
        b = np.mean(kw) - a * np.mean(vw)
        kihat = v[i]*a + b
        correction[i, i] = k[i] - kihat

    K = K - correction
    # ZZ is then recentred
    K = H.T @ K @ H
    return K


def kpls(K, Y, ncp=None, penalization=False, err_lim=1e-9, nloop_max=200, warning_tag=True):
    n = K.shape[0]
    py = Y.shape[1]
    # Array initialisation
    if ncp is None:
        ncp = np.linalg.matrix_rank(K)
    T = np.zeros((n,ncp))
    U = np.zeros((n,ncp))
    C = np.zeros((py,ncp))
    nc = 0
    u = Y[:,0]
    nloop = 0
    warning = None
    Kcorr = None
    while nc < ncp:
        nc += 1
        err = np.inf
        while err > err_lim and nloop < nloop_max:
            if nc < 2:
                if penalization:
                    K = diagonal_correction(K, u)
                    Kcorr = K
            t = K @ u
            t = t / np.sqrt(np.sum(t**2))
            c = Y.T @ t
            u_new = Y @ c
            u_new = u_new/np.sqrt(np.sum(u_new**2))
            err = np.sum((u - u_new)**2)
            u = u_new
            nloop += 1

        if nloop < nloop_max or nc == 1:
            # Deflation of K and Y
            K = K - np.outer(t, t) @ K - K @ np.outer(t, t) + np.outer(t, t) @ K @ np.outer(t, t)
            Y = Y - np.outer(t, t) @ Y

            try:
                T[:, nc-1] = t[:, 0]
                U[:, nc-1] = u[:, 0]
                C[:, nc-1] = c
            except:
                T[:, nc-1] = t
                U[:, nc-1] = u
                C[:, nc-1] = c

            if nloop >= nloop_max:
                T = np.delete(T, np.s_[nc:], 1)
                U = np.delete(U, np.s_[nc:], 1)
                C = np.delete(C, np.s_[nc:], 1)
                if warning_tag:
                    warning = "Component " + str(nc) + " has not converged."
                break
        else:
            T = np.delete(T, np.s_[nc-1:], 1)
            U = np.delete(U, np.s_[nc-1:], 1)
            C = np.delete(C, np.s_[nc-1:], 1)

            break

    return T, U, C, Kcorr, warning




