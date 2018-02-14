from abc import ABCMeta, abstractmethod
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.stats import norm
from progressbar import ProgressBar


def bs(t, S, K, T, sigma, r, div=0):
    tau = T - t
    rexp, dexp = np.exp(-r*tau), np.exp(-div*tau)
    d1 = (np.log(S/K) + (r-div+0.5*sigma**2)*tau) / (
        sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    call = S*dexp*norm.cdf(d1) - K*rexp*norm.cdf(d2)
    put = K*rexp - S*dexp + call
    return call, put


def bs_digital(t, S, K, T, sigma, r ,div=0):
    tau = T - t
    rexp, dexp = np.exp(-r*tau), np.exp(-div*tau)
    d2 = (np.log(S/K) + (r-div-0.5*sigma**2)*tau) / (
        sigma*np.sqrt(tau))
    call = rexp*norm.cdf(d2)
    put = rexp - call
    return call, put


def bs_down_in(t, S, K, H, T, sigma, r, div=0):
    tau = T - t
    rexp, dexp = np.exp(-r*tau), np.exp(-div*tau)
    lam = (r-div+0.5*sigma**2)/(sigma**2)
    y1 = np.log(H**2/(S*K)) / (
        sigma*np.sqrt(tau)) + lam*sigma*np.sqrt(tau)
    y2 = y1-sigma*np.sqrt(tau)
    call = S*dexp*((H/S)**(2*lam))*norm.cdf(y1) - (
        K*rexp*((H/S)**(2*lam-2))*norm.cdf(y2))
    return call


class MonteCarloEstimator(object):
    @staticmethod
    def vanilla_call(K, tau, r, L=None):
        def __payoff(paths, paths_a=None):
            call = np.fmax(
                0, np.exp(-r * tau) * (paths[-1, :] - K))
            if paths_a is not None:
                call += np.fmax(
                    0, np.exp(-r * tau) * (paths_a[-1, :] - K))
                call /= 2
            if L is not None:
                Phi_L = norm.cdf(L)
                call *= (1 - Phi_L)
            return call

        return __payoff

    @staticmethod
    def vanilla_call_parity(K, S0, tau, r):
        def __payoff(paths, paths_a=None):
            put = np.fmax(
                0, np.exp(-r * tau) * (K - paths[-1, :]))
            return put + S0 - K * np.exp(-r * tau)

        return __payoff

    @staticmethod
    def arithmetic_asian_call(K, tau, r):
        def __payoff(paths, paths_a=None):
            call = np.fmax(
                0, np.exp(-r * tau) * (np.mean(paths, axis=0) - K))
            if paths_a is not None:
                call += np.fmax(0, np.exp(-r * tau) * (
                    np.mean(paths_a, axis=0) - K))
                call /= 2
            return call

        return __payoff

    @staticmethod
    def down_in_digital(K, H, tau, r,
                        stop=False,
                        sigma=None, div=0):
        def __payoff(paths):
            pay = 10000.0 * (paths[-1, :] > K) * (
                np.min(paths, axis=0) < H)
            pay *= np.exp(-r * tau)
            return pay

        def __conditional_mc_payoff(paths,
                                    stopping_times,
                                    rn_derivatives=None):
            n_steps, size = paths.shape
            dt = tau / n_steps
            pay = np.zeros(size)
            for j, stop in enumerate(stopping_times):
                if stop < 0:
                    continue
                else:
                    t = stop * dt
                    pay[j] = np.exp(-r * t) * bs_digital(
                        0, paths[stop - 1, j],
                        K, tau - t, sigma, r, div=0)[0] * 10000.0
            return pay

        if stop: return __conditional_mc_payoff
        return __payoff

    @staticmethod
    def down_in_call(K, H, tau, r,
                     stop=False,
                     sigma=None,
                     div=0, theta=0):
        def __payoff(paths):
            pay = np.fmax(
                0, (paths[-1, :] - K)) * (
                      np.min(paths, axis=0) < H)
            pay *= np.exp(-r * tau)
            return pay

        def __conditional_mc_payoff(paths,
                                    stopping_times,
                                    rn_derivatives=None):
            n_steps, size = paths.shape
            dt = tau / n_steps
            pay = np.zeros(size)
            for j, stop in enumerate(stopping_times):
                if stop < 0:
                    continue
                else:
                    t = stop * dt
                    pay[j] = np.exp(-r * t) * bs(
                        0, paths[stop - 1, j],
                        K, tau - t, sigma, r, div=0)[0]
                    if rn_derivatives is not None:
                        pay[j] *= rn_derivatives[j]
            return pay

        if stop: return __conditional_mc_payoff
        return __payoff

    @staticmethod
    def gbm(S0, T, r, sigma, div=0,
            antithetic=False, truncate=None,
            stop=None, theta=0):
        def __gbm(n_steps, size):
            dt = T / n_steps
            print("[gbm]: Initializing grids...")
            time.sleep(0.5)
            S = np.random.normal(size=(n_steps, size))
            bar = ProgressBar()
            for j in bar(range(size)):
                z = S[:, j]
                logr = np.cumsum(sigma * np.sqrt(dt) * z + (
                    r - div - 0.5 * sigma ** 2) * dt)
                S[:, j] = S0 * np.exp(logr)
            return S,

        def __antithetic_gbm(n_steps, size):
            dt = T / n_steps
            print("[antithetic gbm]: Initializing grids...")
            time.sleep(0.5)
            S = np.random.normal(size=(n_steps, size))
            S_a = -S
            bar = ProgressBar()
            for j in bar(range(size)):
                z = S[:, j];
                z_a = S_a[:, j]
                logr = np.cumsum(sigma * np.sqrt(dt) * z + (
                    r - div - 0.5 * sigma ** 2) * dt)
                logr_a = np.cumsum(sigma * np.sqrt(dt) * z_a + (
                    r - div - 0.5 * sigma ** 2) * dt)
                S[:, j] = S0 * np.exp(logr)
                S_a[:, j] = S0 * np.exp(logr_a)
            return S, S_a,

        def __truncated_gbm(n_steps, size):
            # Importance sampling
            # truncate = K
            dt = T / n_steps
            L = (np.log(truncate / S0) - (
                r - div - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            Phi_L = norm.cdf(L)
            print("[truncated gbm]: warning ",
                  "only final prices generated...")
            X = np.random.uniform(
                size=(2, size)) * (1 - Phi_L) + Phi_L
            X = norm.ppf(X)
            x = X[-1, :]
            logr = sigma * np.sqrt(T) * x + (
                                                r - div - 0.5 * sigma ** 2) * T
            X[-1, :] = S0 * np.exp(logr)
            return X,

        def __stopped_gbm(n_steps, size):
            dt = T / n_steps
            print("[stopped gbm]: Initializing grids...")
            time.sleep(0.5)
            stopping_times = -np.ones(size, dtype=np.int8)
            rn_derivatives = np.ones(size)
            S = np.zeros((n_steps + 1, size))
            S[0, :] = np.ones(size) * S0
            bar = ProgressBar()
            for j in bar(range(size)):
                for t in range(1, n_steps + 1):
                    z = np.random.normal(
                        loc=theta)
                    logr = sigma * np.sqrt(dt) * z + (
                                                         r - div - 0.5 * sigma ** 2) * dt
                    S[t, j] = S[t - 1, j] * np.exp(logr)
                    if stop(t, S[t, j]):
                        stopping_times[j] = t
                        if theta != 0:
                            rn_derivatives[j] *= np.exp(
                                (-theta * z) + (0.5 * theta ** 2))
                        break
            return S[1:, :], stopping_times, rn_derivatives

        if stop is not None: return __stopped_gbm
        if truncate is not None: return __truncated_gbm
        return __antithetic_gbm if antithetic else __gbm

    @staticmethod
    def control_S_T(S0, tau, r):
        def __control(paths, paths_a=None):
            control = paths[-1, :]
            if paths_a is not None:
                control += paths_a[-1, :]
                control /= 2
            gbm_mean = S0 * np.exp(r * tau)
            adj = np.mean(control) - gbm_mean
            return control, adj

        return __control

    @staticmethod
    def control_geometric_asian_call(
            S0, K, tau, sigma, r, div=0, n_steps=None):
        def __control(paths, paths_a=None):
            control = np.fmax(
                0, np.exp(-r * tau) * (
                    stats.mstats.gmean(paths, axis=0) - K))
            if paths_a is not None:
                control += np.fmax(
                    0, np.exp(-r * tau) * (
                        stats.mstats.gmean(paths, axis=0) - K))
                control /= 2
            if n_steps is None:
                # asymptotic result
                sigma_star = sigma / np.sqrt(3)
                div_star = ((r + div) / 2) + (sigma ** 2 / 12)
            else:
                N = n_steps
                sigma_star = sigma * np.sqrt(
                    (N + 1) * (2 * N + 1) / (6 * N ** 2))
                div_star = r * ((N - 1) / (2 * N)) + div * (
                (N + 1) / (2 * N)) + (
                               sigma ** 2 * ((N + 1) * (N - 1) / (12 * N ** 2)))
            geom_asian_mean = bs(
                0, S0, K, tau, sigma_star, r, div_star)[0]
            adj = np.mean(control) - geom_asian_mean
            return control, adj

        return __control

    def __init__(self, sampler=None, payoff=None, control=None):
        self.path_sampler = sampler
        self.payoff = payoff
        self.control = control

    def estimate(self, n_steps, n_size):
        assert (self.path_sampler and self.payoff)
        sde_paths = self.path_sampler(n_steps, n_size)
        sample = self.payoff(*sde_paths)
        sample_mean, se = np.mean(sample), stats.sem(sample, ddof=0)
        if self.control is not None:
            control, adj = self.control(*sde_paths)
            cov_xy = np.cov(control, sample)
            rho = np.corrcoef(control, sample)[0, 1]
            a_hat = -cov_xy[0, 1] / cov_xy[0, 0]
            sample_mean += a_hat * adj
            se *= np.sqrt(1 - rho ** 2)
        return sample, sample_mean, se

    def reset_models(self, sampler=None, payoff=None, control=None):
        if sampler:
            self.path_sampler = sampler
        if payoff:
            self.payoff = payoff
        if control:
            self.control = control
