import numpy as np


def transprob(M, s, nu, rng):
    P = np.zeros((M, M))
    T = len(s)
    nt = np.array([s[:T-1], s[-T+1:]])
    transnum = np.zeros((M, M))

    for im in range(M):
        for j in range(M):
            nt0 = nt[0] == im
            nt1 = nt[1] == j
            transnum[im, j] = np.sum(nt0 & nt1)
        P[im] = rng.dirichlet(nu[im] + transnum[im])
    # print(transnum)
    return P
