from Posterior import *
import numpy as np


def Gibbs_MS_Sampler(Y, X, rank, nmod, nobs, j, qm, p_tau, a_tau, a_lamb, blam, p_sig, a_sig, beta, zeta, tau, w, sigma_m, sigma_k, mu, d_sel, m_sel, I_0, rng, alpha=1, bfixed=False):
    # Construct coefficients B from beta
    bkd = np.zeros(j)
    bkd_transform = [np.array([np.swapaxes(bkd, 0, m) for d in range(rank)]) for m in range(nmod)]
    Bkd = np.array([[np.swapaxes([beta[m][d, jm].reshape(bkd_transform[m][d, jm].shape) for jm in range(j[m])], 0, m) for m in range(nmod)] for d in range(rank)])
    Bd = np.prod(Bkd, axis=1)

    psi_out_sum, psi, psi_out = psi_initial(X, Bkd, rank, nmod, nobs, j, qm)

    psi_ytil_sum, psi_ytil, Rdt, Rjt, ytil = residual_initial(Y, X, Bd, psi, rank, nmod, nobs, j)

    for d in d_sel:
        for m in m_sel:
            for jm in range(j[m]):
                beta_cand = p_beta_given_rest(tau, zeta[d], w[m][d, jm], sigma_m[m], sigma_k, psi_out_sum[m][d, jm], psi_ytil_sum[m][d, jm], qm[m], rng, bfixed=bfixed)
                if max(abs(beta_cand)) <= 6:
                    beta[m][d, jm] = beta_cand
                    np.swapaxes(Bkd[d, m], 0, m)[jm] = beta_cand.reshape(np.swapaxes(Bkd[d, m], 0, m).shape[1:])
                    Bd[d] = np.prod(Bkd[d], axis=0)

                    # Update Rjt, y_tilde
                    for t in range(nobs):
                        for _ in range(j[m]):
                            if _ != jm:
                                X_r = X[t].copy()
                                np.swapaxes(X_r, 0, m)[_] = 0
                                Rjt[m][t, d, _] = np.sum(Bd[d] * X_r)
                                if bfixed:
                                    ytil[m][t, d, _] = Y[t] - Rjt[m][t, d, _] - Rdt[t, d] - mu[t]
                                    psi_ytil[m][t, d, _] = psi[m][t, d, _] * ytil[m][t, d, _] / sigma_k[t]
                                else:
                                    ytil[m][t, d, _] = Y[t] - Rjt[m][t, d, _] - Rdt[t, d] - mu
                                    psi_ytil[m][t, d, _] = psi[m][t, d, _] * ytil[m][t, d, _]
                    psi_ytil_sum[m][d] = psi_ytil[m][:, d].sum(axis=0)
            # Update psi for each k (mode)
            psi_out_sum, psi, psi_out = psi_update(X, Bkd, psi, psi_out, psi_out_sum, nmod, nobs, d, m, j, sigma_k, bfixed=bfixed)

        # Update Rdt, cd for each d (rank)
        for t in range(nobs):
            for _ in range(rank):
                if _ != d:
                    Bd_r = Bd.copy()
                    Bd_r[_] = 0
                    Rdt[t, _] = np.vdot(np.sum(Bd_r, axis=0), X[t])

    B = np.sum(Bd, axis=0)

    gamma = [np.array([[p_gamma_given_rest(beta[m][d, jm], tau, zeta[d], w[m][d, jm], sigma_m[m], qm[m], rng) for jm in range(j[m])] for d in range(rank)]) for m in range(nmod)]

    beta_gamma, beta_gamma_sum = beta_gamma_update(beta, gamma, w, sigma_m, rank, nmod, j)

    cd = np.array([beta_gamma_sum[m].sum(axis=1) for m in range(nmod)]).sum(axis=0)

    zeta = p_zeta_given_rest(I_0, alpha, cd, tau, rank, rng)

    tau = p_tau_given_rest(p_tau, a_tau, cd, zeta, rank, rng)

    lamb = p_lambda_given_rest(gamma, a_lamb, tau, zeta, blam, rank, nmod, rng)

    w = p_w_given_rest(gamma, lamb, tau, zeta, rank, nmod, j, rng)

    sigma_m = p_sigma_m_given_rest(p_sig, a_sig, beta_gamma, zeta, tau, rank, nmod, rng)

    return B, beta, gamma, zeta, tau, lamb, w, sigma_m
