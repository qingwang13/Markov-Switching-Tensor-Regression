import numpy as np


def allocation(N, M, nitr, rank, nmod, p, qk, bfixed=False, multi=False, L=0):
    if bfixed:
        BGibbs = np.array([np.zeros(p) for _ in range(nitr)], dtype='float')
        sGibbs = np.zeros((nitr, N), dtype=int)
        sigma_kGibbs = np.ones((nitr, N))  # variance is considered time-specific when b is fixed
        gammaGibbs = [[np.array([[.001 for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for i in range(nitr)]
        betaGibbs = [[np.array([[.001 * np.ones(qk[k]) for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for i in range(nitr)]
        zetaGibbs = np.array([[1 / rank for _ in range(rank)] for i in range(nitr)])
        tauGibbs = np.ones(nitr) * .1
        lambGibbs = np.ones((nitr, nmod, rank))
        wGibbs = [[np.array([[1. for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for i in range(nitr)]
        sigma_mGibbs = np.ones((nitr, nmod)) * .01
        PGibbs = np.zeros((nitr, M, M))
        filtp = np.ones((nitr, N, M)) * .5
    elif multi:
        BGibbs = np.array([[[np.zeros(p) for _ in range(L)] for _ in range(M)] for _ in range(nitr)], dtype='float')
        sGibbs = np.zeros((nitr, N), dtype=int)
        sigma_kGibbs = np.ones((nitr, M, L))
        gammaGibbs = [[[[np.array([[.001 for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for l in range(L)] for m in range(M)] for i in range(nitr)]
        betaGibbs = [[[[np.array([[.001 * np.ones(qk[k]) for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for l in range(L)] for m in range(M)] for i in range(nitr)]
        zetaGibbs = np.array([[[[1 / rank for _ in range(rank)] for l in range(L)] for m in range(M)] for i in range(nitr)])
        tauGibbs = np.ones((nitr, M, L)) * .1
        lambGibbs = np.ones((nitr, M, L, nmod, rank))
        wGibbs = [[[[np.array([[1. for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for l in range(L)] for m in range(M)] for i in range(nitr)]
        sigma_mGibbs = np.ones((nitr, M, L, nmod)) * .01
        PGibbs = np.zeros((nitr, M, M))
        filtp = np.ones((nitr, N, M)) * .5
    else:
        BGibbs = np.array([[np.zeros(p) for _ in range(M)] for _ in range(nitr)], dtype='float')
        sGibbs = np.zeros((nitr, N), dtype=int)
        sigma_kGibbs = np.ones((nitr, M))
        gammaGibbs = [[[np.array([[.001 for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for m in range(M)] for i in range(nitr)]
        betaGibbs = [[[np.array([[.001 * np.ones(qk[k]) for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for m in range(M)] for i in range(nitr)]
        zetaGibbs = np.array([[[1 / rank for _ in range(rank)] for m in range(M)] for i in range(nitr)])
        tauGibbs = np.ones((nitr, M)) * .1
        lambGibbs = np.ones((nitr, M, nmod, rank))
        wGibbs = [[[np.array([[1. for _ in range(p[k])] for d in range(rank)]) for k in range(nmod)] for m in range(M)] for i in range(nitr)]
        sigma_mGibbs = np.ones((nitr, M, nmod)) * .01
        PGibbs = np.zeros((nitr, M, M))
        filtp = np.ones((nitr, N, M)) * .5
    return BGibbs, sGibbs, sigma_kGibbs, gammaGibbs, betaGibbs, zetaGibbs, tauGibbs, lambGibbs, wGibbs, sigma_mGibbs, PGibbs, filtp