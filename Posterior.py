import numpy as np
from gigrnd import gigrnd

rng = np.random.default_rng(seed=100)


def p_zeta_given_rest(I0, alpha, cd, tau, rank, rng):
    a = I0 / 2 - alpha / rank
    b = np.array([cd[_] / tau for _ in range(rank)])
    phi = np.array([1 / rng.gamma(a, b[_] ** -1) for _ in range(rank)])
    zeta = np.array([phi[_] / np.sum(phi) for _ in range(rank)])
    return zeta


def p_tau_given_rest(p_tau, a_tau, cd, zeta, rank, rng):
    b_tau = np.sum(cd / zeta)
    return gigrnd(p_tau, a_tau, b_tau, rng)


def p_lambda_given_rest(gamma, a_lamb, tau, zeta, blam, rank, nmod, rng):
    lamb = np.zeros((nmod, rank))
    for d in range(rank):
        for k in range(nmod):
            b_lamb = np.sum(abs(gamma[k][d])) / (tau * zeta[d]) ** .5 + blam
            lamb[k, d] = rng.gamma(a_lamb[k], b_lamb ** -1)
    return lamb


def p_w_given_rest(gamma, lamb, tau, zeta, rank, nmod, j, rng):
    w = [np.array([[.0001 for _ in range(j[k])] for d in range(rank)]) for k in range(nmod)]
    for d in range(rank):
        for k in range(nmod):
            for jk in range(j[k]):
                a_w = lamb[k, d] ** 2
                b_w = gamma[k][d, jk] ** 2 / (tau * zeta[d])
                if b_w == 0:
                    w[k][d, jk] = rng.gamma(.5, (a_w * .5) ** -1)
                else:
                    w[k][d, jk] = gigrnd(.5, a_w, b_w, rng)
    return w


def p_w_given_rest2(gamma, lamb, tau, zeta, rng):
    a_w = lamb ** 2
    b_w = gamma ** 2 / (tau * zeta)
    return gigrnd(.5, a_w, b_w, rng)


def p_beta_given_rest(tau, zeta, w, sigma_m, sigma_k, psi_out, psi_yt, qk, rng, bfixed=False):
    xi = (tau * zeta * (w + sigma_m)) ** -1 * np.identity(int(qk))
    if bfixed:
        var_beta = np.linalg.inv(psi_out + xi)
        mean_beta = var_beta @ psi_yt
    else:
        var_beta = np.linalg.inv(psi_out / sigma_k + xi)
        mean_beta = var_beta @ psi_yt / sigma_k
    beta = rng.multivariate_normal(mean_beta, var_beta)
    return beta


#  posterior of beta without integrating out gamma
def p_beta_given_rest2(tau, zeta, sigma_m, sigma_k, psi_out, psi_yt, qk, rng, gamma=0, bfixed=False):
    xi = (tau * zeta * sigma_m) ** -1
    if bfixed:
        var_beta = np.linalg.inv(psi_out + xi * np.identity(int(qk)))
        mean_beta = var_beta @ (psi_yt + xi * gamma)
    else:
        var_beta = np.linalg.inv(psi_out / sigma_k + xi * np.identity(int(qk)))
        mean_beta = var_beta @ (psi_yt / sigma_k + xi * gamma)
    beta = rng.multivariate_normal(mean_beta, var_beta)
    return beta


def p_gamma_given_rest(beta, tau, zeta, w, sigma_k, qk, rng):
    mean_gamma = w / (qk * w + sigma_k) * np.sum(beta)
    var_gamma = tau * zeta * w * sigma_k / (qk * w + sigma_k)
    return rng.normal(mean_gamma, var_gamma ** .5)


def p_sigma_m_given_rest(p_sig, a_sig, beta_gamma, zeta, tau, rank, nmod, rng):
    sigma_k = np.array([.0001 for _ in range(nmod)])
    b_sig = np.array([[beta_gamma[k].sum(axis=1)[d] / zeta[d] for d in range(rank)] for k in range(nmod)]).sum(axis=1)
    for k in range(nmod):
        sigma_k[k] = gigrnd(p_sig, a_sig, b_sig[k] / tau, rng)
    return sigma_k


def p_mu_given_rest(y, x, B, sig_y, sig_mu, rng):
    var_mu = (len(y) / sig_y + 1 / sig_mu) ** (-1)
    mean_mu = var_mu * (np.sum(y - x.reshape(x.shape[0], -1) @ B.flatten()) / sig_y)
    return rng.normal(mean_mu, var_mu ** .5)


def p_sigma_k_given_rest(a_sig, b_sig, y, x, b, mu, rng):
    nobs = len(y)
    asig_bar = a_sig + nobs / 2
    bsig_bar = b_sig + np.sum((y - x.reshape(x.shape[0], -1) @ b.flatten() - mu) ** 2) / 2
    return 1 / rng.gamma(asig_bar, bsig_bar ** -1)


def psi_initial(X, Bkd, rank, nmod, nobs, j, qk):
    psi = [np.array([[[np.zeros(qk[k]) for _ in range(j[k])] for d in range(rank)] for t in range(nobs)]) for k in range(nmod)]
    for t in range(nobs):
        for d in range(rank):
            for k in range(nmod):
                Bkd_r = Bkd[d].copy()
                Bkd_r[k] = 1
                hp = np.prod(Bkd_r, axis=0) * X[t]
                for jk in range(j[k]):
                    psi[k][t, d, jk] = np.swapaxes(hp, 0, k)[jk].flatten()
    psi_out = [
        np.array([[[np.outer(psi[k][t, d, jk], psi[k][t, d, jk]) for jk in range(j[k])] for d in range(rank)] for t in range(nobs)]) for k in
         range(nmod)]
    psi_out_sum = [psi_out[m].sum(axis=0) for m in range(nmod)]
    return psi_out_sum, psi, psi_out


def psi_update(X, Bkd, psi, psi_out, psi_out_sum, nmod, nobs, d, k, j, sigmak, bfixed=False):
    for i in range(nmod):
        if i != k:
            Bkd_r = Bkd[d].copy()
            Bkd_r[i] = 1
            for t in range(nobs):
                hp = np.prod(Bkd_r, axis=0) * X[t]
                for jk in range(j[i]):
                    psi[i][t, d, jk] = np.swapaxes(hp, 0, i)[jk].flatten()
                    if bfixed:
                        psi_out[i][t, d, jk] = np.outer(psi[i][t, d, jk], psi[i][t, d, jk]) / sigmak[t]
                    else:
                        psi_out[i][t, d, jk] = np.outer(psi[i][t, d, jk], psi[i][t, d, jk])
            psi_out_sum[i][d] = psi_out[i][:, d].sum(axis=0)

    return psi_out_sum, psi, psi_out


def residual_initial(Y, X, Bd, psi, rank, nmod, nobs, j):
    Rdt = np.array([[.001 for _ in range(rank)] for _ in range(nobs)])

    Rjt = [np.array([[[.001 for _ in range(j[k])] for d in range(rank)] for t in range(nobs)]) for k in range(nmod)]

    for t in range(nobs):
        for d in range(rank):
            Bd_r = np.copy(Bd)
            Bd_r[d] = 0
            Rdt[t, d] = np.vdot(np.sum(Bd_r, axis=0), X[t])
            for m in range(nmod):
                for jm in range(j[m]):
                    X_r = np.copy(X[t])
                    np.swapaxes(X_r, 0, m)[jm] = 0
                    Rjt[m][t, d, jm] = np.sum(Bd[d] * X_r)

    ytil = [np.array([[[Y[t] - Rjt[k][t, d, jk] - Rdt[t, d] for jk in range(j[k])] for d in range(rank)] for t in range(nobs)]) for k in range(nmod)]

    psi_ytil = [np.array([[[psi[k][t, d, jk] * ytil[k][t, d, jk] for jk in range(j[k])] for d in range(rank)] for t in range(nobs)]) for k in range(nmod)]

    psi_ytil_sum = [psi_ytil[m].sum(axis=0) for m in range(nmod)]

    return psi_ytil_sum, psi_ytil, Rdt, Rjt, ytil


def beta_gamma_update(beta, gamma, w, sigma_k, rank, nmod, j):
    beta_gamma = [np.array([[np.sum((beta[k][d, jk] - gamma[k][d, jk])**2) for jk in range(j[k])] for d in range(rank)]) for k
                  in range(nmod)]
    beta_gamma_sum = [np.array([[beta_gamma[k][d, jk] / sigma_k[k] + gamma[k][d, jk] ** 2 / w[k][d, jk] for jk in
                                 range(j[k])] for d in range(rank)]) for k in range(nmod)]
    return beta_gamma, beta_gamma_sum


def flip_list(main_list, index1=0, index2=1):
    main_list[index1], main_list[index2] = main_list[index2], main_list[index1]
    return main_list


def rs_components(comp, mode, iter, disc=2, nfs=10):  # specify the rank, mode, # of iterations, discount factor and of full scan
    full_d = np.arange(comp)
    full_m = np.arange(mode)
    sel_d = max(2, comp // disc)
    sel_m = max(2, mode // disc)

    d_sel_1 = [full_d for _ in range(nfs)]
    m_sel_1 = [full_m for _ in range(nfs)]

    d_sel_2 = [rng.choice(comp, sel_d, replace=False) for _ in range(iter - nfs)]
    m_sel_2 = [rng.choice(mode, sel_m, replace=False) for _ in range(iter - nfs)]
    return d_sel_1+d_sel_2, m_sel_1+m_sel_2
