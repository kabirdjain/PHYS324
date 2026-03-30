import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
from alerce.core import Alerce


# better than rbf for supernovae?
def matern32(t1, t2, length_scale, amplitude):
    r = np.abs(t1[:, None] - t2[None, :]) / length_scale
    return amplitude**2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)

# most likely hetero gp
def log_noise_gp(t, log_var_obs, length_scale=0.3, amplitude=1.0, noise_floor=1e-4):
    K = matern32(t, t, length_scale, amplitude)
    K += noise_floor * np.eye(len(t))
    L = np.linalg.cholesky(K + 1e-6 * np.eye(len(t)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, log_var_obs))
    return K, L, alpha

class HeteroscedasticGP:
    """
    Use EM algo to approximate noise levels (this is just most likely hetero)
    """
    def __init__(self, n_iter=3):
        self.n_iter = n_iter

    def fit(self, t, y, init_params=None):
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        init_ls = init_params[0] if init_params else 0.2
        init_amp = init_params[1] if init_params else 1.0
        init_noise = init_params[2] if init_params else 0.05

        noise_var = np.full(len(t), init_noise**2)

        for _ in range(self.n_iter):
            def neg_log_marginal(params):
                ls, amp = np.exp(params[0]), np.exp(params[1])
                K = matern32(t, t, ls, amp)
                Ky = K + np.diag(noise_var)
                try:
                    L = np.linalg.cholesky(Ky)
                except np.linalg.LinAlgError:
                    return 1e10
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                return (0.5 * y @ alpha
                        + np.sum(np.log(np.diag(L)))
                        + 0.5 * len(t) * np.log(2 * np.pi))

            # bfgs = goat optimizer
            res = minimize(neg_log_marginal, [np.log(init_ls), np.log(init_amp)],
                           method='L-BFGS-B',
                           bounds=[(-4, 1), (-3, 2)])
            ls, amp = np.exp(res.x[0]), np.exp(res.x[1])

            K = matern32(t, t, ls, amp)
            Ky = K + np.diag(noise_var)
            L = np.linalg.cholesky(Ky + 1e-8 * np.eye(len(t)))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            mu = K @ alpha
            residuals = y - mu

            log_sq_resid = np.log(residuals**2 + 1e-6)
            _, _, noise_alpha = log_noise_gp(t, log_sq_resid)
            noise_K = matern32(t, t, 0.15, 1.0)
            noise_mu = noise_K @ noise_alpha
            noise_var = np.exp(noise_mu) + 1e-6

        self.ls_ = ls
        self.amp_ = amp
        self.t_train_ = t
        self.y_train_ = y
        self.noise_var_ = noise_var
        self.L_ = L
        self.alpha_ = alpha
        self.K_train_ = K
        return self

    def predict(self, t_test):
        t_test = np.asarray(t_test, dtype=float)
        K_s = matern32(t_test, self.t_train_, self.ls_, self.amp_)
        K_ss = matern32(t_test, t_test, self.ls_, self.amp_)

        mu = K_s @ self.alpha_
        v = np.linalg.solve(self.L_, K_s.T)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)

        noise_K = matern32(t_test, self.t_train_, 0.15, 1.0)
        log_sq_resid = np.log(self.noise_var_)
        _, _, noise_alpha = log_noise_gp(self.t_train_, log_sq_resid)
        noise_mu = noise_K @ noise_alpha
        noise_var_test = np.exp(noise_mu) + 1e-6

        return mu, np.sqrt(np.maximum(var, 0)), np.sqrt(noise_var_test)

    def log_likelihood(self, t_new, y_new):
        """Log p(y_new | GP posterior) — lower = more anomalous"""
        mu, sig_gp, sig_noise = self.predict(t_new)
        total_var = sig_gp**2 + sig_noise**2
        return -0.5 * np.sum((y_new - mu)**2 / total_var + np.log(total_var))

def fit_all_gps(curves_data):
    fitted = []
    for oid, t, y, raw_t, raw_y in tqdm(curves_data, desc="Fitting GPs"):
        try:
            gp = HeteroscedasticGP(n_iter=3).fit(t, y)
            fitted.append((oid, gp, t, y))
        except Exception as e:
            tqdm.write(f"Skipping {oid}: {e}")
    return fitted

def build_template(fitted_gps, n_grid=200):
    """
    Evaluate all fitted GPs on a common time grid.
    Template = mean of individual posterior means.
    Template uncertainty = std across curves.
    """
    t_grid = np.linspace(0, 1, n_grid)
    all_means = []
    all_noise_vars = []

    for oid, gp, t, y in fitted_gps:
        mu, sig_gp, sig_noise = gp.predict(t_grid)
        all_means.append(mu)
        all_noise_vars.append(sig_noise**2)

    all_means = np.array(all_means)      # (n_curves, n_grid)
    all_noise = np.array(all_noise_vars)  # (n_curves, n_grid)

    template_mu = np.mean(all_means, axis=0)
    template_std = np.std(all_means, axis=0)  # population spread
    mean_noise = np.mean(all_noise, axis=0)   # avg heteroscedastic noise

    return t_grid, template_mu, template_std, mean_noise, all_means, all_noise

def score_precursor(gp, t_obs, y_obs, t_grid, template_mu, template_std,
                    pre_peak_frac=0.3, sigma_thresh=2.5):
    """
    Check window before max peak, find anomalies there by comparing anomaly gp to total gp distro.
    """
    mu, sig_gp, sig_noise = gp.predict(t_grid)

    peak_idx = np.argmin(mu)
    peak_t = t_grid[peak_idx]

    pre_mask = t_grid < peak_t * pre_peak_frac
    if pre_mask.sum() < 3:
        pre_mask = t_grid < 0.15  # fallback: first 15%

    deviation = np.abs(mu[pre_mask] - template_mu[pre_mask])
    normalized_dev = deviation / (template_std[pre_mask] + 1e-6)
    mean_deviation_score = np.mean(normalized_dev)

    pre_noise = sig_noise[pre_mask].mean()
    post_mask = t_grid > peak_t * 1.2
    post_noise = sig_noise[post_mask].mean() if post_mask.sum() > 3 else sig_noise.mean()
    noise_excess = pre_noise / (post_noise + 1e-6)

    # log-likelihood of observed pre-peak points under template
    pre_obs_mask = t_obs < peak_t * pre_peak_frac
    if pre_obs_mask.sum() >= 2:
        t_pre = t_obs[pre_obs_mask]
        y_pre = y_obs[pre_obs_mask]
        tmpl_mu_obs = np.interp(t_pre, t_grid, template_mu)
        tmpl_std_obs = np.interp(t_pre, t_grid, template_std)
        ll_score = -np.mean((y_pre - tmpl_mu_obs)**2 / (tmpl_std_obs**2 + 1e-6))
    else:
        ll_score = 0.0

    is_precursor = (mean_deviation_score > sigma_thresh) or (noise_excess > 2.0)

    return {
        'peak_t': peak_t,
        'mean_deviation_score': mean_deviation_score,
        'noise_excess_ratio': noise_excess,
        'pre_peak_ll': ll_score,
        'is_precursor': is_precursor,
    }

