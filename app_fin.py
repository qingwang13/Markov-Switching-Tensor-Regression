import numpy as np
import pickle
import os
import time
import copy
from joblib import Parallel, delayed
from allocation import allocation
from transprob import transprob
from ffbs import ffbs
from Gibbs_MS_Sampler import Gibbs_MS_Sampler
from datetime import datetime
from Posterior import p_mu_given_rest, rs_components

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

rng = np.random.default_rng(seed=100)


def moving_average(series, window_size):
    return np.convolve(series, np.ones(window_size) / window_size, mode='valid')


def s_ini(resp, reg, n_states, n_reg, wins=40):  # n_states: number of hidden states, n_reg: number of regressions
    reg_vec = reg.reshape(reg.shape[0], -1).copy()
    b_ols = np.linalg.inv(reg_vec.T @ reg_vec) @ reg_vec.T @ resp
    Y_pred = reg_vec @ b_ols
    err_sq = (resp - Y_pred) ** 2
    err_sq_ma = moving_average(err_sq, wins)
    err_sq[wins - 1:] = err_sq_ma
    if n_states == 2:
        thr = np.quantile(err_sq_ma, .45)
        s = err_sq > thr
        s[:wins] = int(0)
        sigma0 = np.mean(err_sq[~s])
        sigma1 = np.mean(err_sq[s])
        sigma = np.array([[sigma0, sigma1] for _ in range(n_reg)]).T
        return s.astype(int), sigma
    else:
        s = np.zeros_like(err_sq)
        quantiles = np.quantile(err_sq_ma, np.linspace(0, 1, n_states + 1)[1:-1])
        for i, q in enumerate(quantiles):
            s[wins:][err_sq[wins:] > q] = i + 1
        sigma = np.array([[np.mean(err_sq[s == i]) for i in range(n_states)] for _ in range(n_reg)]).T
        return s.astype(int), sigma


# load the data for financial application
fin_data = pickle.load(open('/home/qing.wang/rshare/qing.wang/Tensor/app/data/fin_data_app_multi.dat', 'rb'))

# sample size
X = np.array([fin_data['X'][:, 1:, :44], fin_data['X'][:, :-1, :44]])
Y = fin_data['Y']
nobs = X[0].shape[0]
nreg = Y.shape[0]  # number of regressions
ss = .3

X_train = X[:, :int(nobs * ss)].copy()
X_test = X[:, int(nobs * ss):].copy()
Y_train = Y[:, :int(nobs * ss)].copy()
Y_test = Y[:, int(nobs * ss):].copy()

# scale the covariates
mean_x = np.mean(X_train, axis=1)
std_x = np.std(X_train, axis=1)
X_train_scale = np.array([(X_train[_] - mean_x[_]) / std_x[_] for _ in range(nreg)])
X_test_scale = np.array([(X_test[_] - mean_x[_]) / std_x[_] for _ in range(nreg)])

# de-mean the dependent variable
mean_y = np.mean(Y_train, axis=1)
std_y = np.std(Y_train, axis=1)
Y_train_scale = np.array([(Y_train[_] - mean_y[_]) / std_y[_] for _ in range(nreg)])
Y_test_scale = np.array([(Y_test[_] - mean_y[_]) / std_y[_] for _ in range(nreg)])

rank = 2
p = X_train[0, 0].shape
nmod = X_train[0].ndim - 1
qm = np.array([np.prod(p) / p[_] for _ in range(len(p))], dtype=int)
I_0 = np.prod(p) * nmod + np.sum(p)
K = 3  # number of regimes
ntrain = X_train[0].shape[0]

#  Hyper parameters
alpha = 1
C = (alpha / (rank + 1)) / (alpha + 1)
asig_m = .5
bsig_m = 9. * C ** .5
atau = .8
btau = 20 / bsig_m
alam = 3
blam = alam ** .4
p_sig = asig_m - rank * np.prod(p) / 2
a_sig = 2 * bsig_m
p_tau = atau - rank * I_0 / 2
a_tau = 2 * btau
p_phi = alpha / rank - I_0 / 2
a_phi = 2 * btau
a_lamb = np.array([p[k] + alam for k in range(nmod)])
nu = np.ones((K, K))  # Prior for the hidden states
phi = np.array([.0001 for _ in range(rank)])
a_sig_k = 5.  # hyperparameter for variances of regimes
b_sig_k = .3
sig_mu = 1

#  MCMC setting
nitr = 1000
burnin = int(nitr * .5)
dur = 8  # minimum duration of a regime, if dur = 1, then no duration constraint, default is 8

# Random Gibbs setting
d_sel, m_sel = rs_components(rank, nmod, nitr)

#  Allocation
BGibbs, sGibbs, sigma_kGibbs, gammaGibbs, betaGibbs, zetaGibbs, tauGibbs, lambGibbs, wGibbs, sigma_mGibbs, PGibbs, filtp \
    = allocation(ntrain, K, nitr, rank, nmod, p, qm, multi=True, L=nreg)
muGibbs = np.zeros((nitr, K, nreg))

#  Initiate the states
sGibbs[0], sigma_kGibbs[0, :] = s_ini(Y_train[0], X_train_scale[0], K, nreg)

#  Gibbs Sampling

func = delayed(Gibbs_MS_Sampler)

st = time.time()
for i in range(1, nitr):
    y_est = np.zeros((K, nreg, ntrain))
    for k in range(K):
        nobs = np.sum(sGibbs[i - 1] == k)
        if nobs == 0:
            BGibbs[i][k] = copy.deepcopy(BGibbs[i - 1][k])
            betaGibbs[i][k] = copy.deepcopy(betaGibbs[i - 1][k])
            gammaGibbs[i][k] = copy.deepcopy(gammaGibbs[i - 1][k])
            zetaGibbs[i][k] = copy.deepcopy(zetaGibbs[i - 1][k])
            tauGibbs[i][k] = copy.deepcopy(tauGibbs[i - 1][k])
            lambGibbs[i][k] = copy.deepcopy(lambGibbs[i - 1][k])
            wGibbs[i][k] = copy.deepcopy(wGibbs[i - 1][k])
            sigma_mGibbs[i][k] = copy.deepcopy(sigma_mGibbs[i - 1][k])
            sigma_kGibbs[i][k] = copy.deepcopy(sigma_kGibbs[i - 1][k])
            y_est[k] = np.array(
                [X_train_scale[l].reshape(ntrain, -1) @ BGibbs[i, k, l].flatten() + muGibbs[i - 1, k][l] for l in
                 range(nreg)])
        else:
            results = (Parallel(n_jobs=nreg)
                       (func(Y_train_scale[_, sGibbs[i - 1] == k], X_train_scale[_, sGibbs[i - 1] == k], rank, nmod,
                             nobs,
                             p, qm, p_tau, a_tau, a_lamb, blam, p_sig, a_sig, betaGibbs[i - 1][k][_],
                             zetaGibbs[i - 1][k][_],
                             tauGibbs[i - 1][k][_], wGibbs[i - 1][k][_], sigma_mGibbs[i - 1][k][_],
                             sigma_kGibbs[i - 1][k][_],
                             muGibbs[i - 1][k][_], d_sel[i - 1], m_sel[i - 1], I_0, rng) for _ in range(nreg)))
            for l, result in enumerate(results):
                BGibbs[i][k][l] = result[0]
                betaGibbs[i][k][l] = result[1]
                gammaGibbs[i][k][l] = result[2]
                zetaGibbs[i][k][l] = result[3]
                tauGibbs[i][k][l] = result[4]
                lambGibbs[i][k][l] = result[5]
                wGibbs[i][k][l] = result[6]
                sigma_mGibbs[i][k][l] = result[7]
                y_est[k, l] = X_train_scale[l].reshape(ntrain, -1) @ BGibbs[i, k, l].flatten() + muGibbs[i - 1, k, l]
                a_sig_k_bar = a_sig_k + nobs / 2
                b_sig_k_bar = b_sig_k + np.sum(
                    (Y_train_scale[l, sGibbs[i - 1] == k] - y_est[k, l][sGibbs[i - 1] == k]) ** 2) / 2
                sigma_kGibbs[i, k, l] = 1 / rng.gamma(a_sig_k_bar, 1 / b_sig_k_bar)
                muGibbs[i, k, l] = p_mu_given_rest(Y_train_scale[l, sGibbs[i - 1] == k],
                                                   X_train_scale[l, sGibbs[i - 1] == k], BGibbs[i, k, l],
                                                   sigma_kGibbs[i, k, l], sig_mu, rng)

    Psim = transprob(K, sGibbs[i - 1], nu, rng)
    PGibbs[i] = Psim
    filtp[i], smooths = ffbs(Y_train_scale, y_est, sigma_kGibbs[i], K, Psim, rng, multi=True, nreg=nreg)
    if dur > 1:
        z = smooths[1:] - smooths[:-1]
        c = 0
        for d in range(2, dur + 1):
            c += np.dot(z[d - 1:], z[:-d + 1])
        while c != 0:
            Psim = transprob(K, sGibbs[i - 1], nu, rng)
            filtp[i], smooths = ffbs(Y_train_scale, y_est, sigma_kGibbs[i], K, Psim, rng, multi=True, nreg=nreg)
            z = smooths[1:] - smooths[:-1]
            c = 0
            for d in range(2, dur + 1):
                c += np.dot(z[d - 1:], z[:-d + 1])
        sGibbs[i] = smooths
        PGibbs[i] = Psim
    else:
        sGibbs[i] = smooths
        PGibbs[i] = Psim
    print('Gibbs iteration: ', i)

end = time.time()
ex_time = (end - st) / 60

now = datetime.now().strftime('%m%d%H%M%S')

output = {'B_post': BGibbs, 'beta_post': betaGibbs, 'gamma_post': gammaGibbs, 'zeta_post': zetaGibbs,
          'tau_post': tauGibbs, 'lambda_post': lambGibbs, 'w_post': wGibbs, 'sigma_m_post': sigma_mGibbs,
          's_post': sGibbs, 'filtp': filtp, 'now': now, 'ex_time': ex_time, 'X_train': X_train_scale,
          'X_test': X_test_scale, 'Y_train': Y_train_scale, 'Y_test': Y_test_scale, 'sigma_k_post': sigma_kGibbs,
          'mu_post': muGibbs, 'P_post': PGibbs, 'dura': dur
          }

file_name = 'MS_fin_' + f'r{rank}k{K}_' + now + '_multi.dat'
save_path = "/home/qing.wang/rshare/qing.wang/Tensor/output/fin/MS/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

pickle.dump(output, open(save_path + file_name, 'wb'))
