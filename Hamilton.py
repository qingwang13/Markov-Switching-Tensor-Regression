import numpy as np


def Hamilton(P, clkl, rng):
    T = clkl.shape[1]
    M = clkl.shape[0]
    filtp = np.zeros((T, M))

    #  Invariant probability
    ip = np.array([*(P.T - np.identity(M)), np.ones(M)])
    if np.linalg.det(ip.T @ ip) != 0:
        ip = np.linalg.inv(ip.T @ ip) @ ip.T
        ip = ip[:, M]
    else:
        ip = np.ones(M) / M

    st1t1 = ip
    for t in range(T):
        stt1 = P.T @ st1t1
        clklmax = clkl[:, t].max()
        stt = stt1 * np.exp(clkl[:, t] - clklmax)
        nc = np.sum(stt)
        stt = stt / nc
        filtp[t] = stt
        st1t1 = stt

    #  Backward Sampling
    smooths = np.zeros(T, dtype='int')
    smooths[T - 1] = rng.multinomial(1, filtp[T - 1]) @ np.arange(M)
    for t in range(T - 2, -1, -1):
        xi = filtp[t] * P[:, smooths[t + 1]]
        xi = xi / np.sum(xi)
        smooths[t] = rng.multinomial(1, xi) @ np.arange(M)

    return filtp, smooths
