from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class BaseRNG(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def draw(self, n=1):
        raise NotImplementedError

    def marginal_plot(self, ax, n_sample, bin_width,
                      truncate=(-np.inf, np.inf),
                      label='_nolegend_', which=None):
        if which is None:
            # uni-variate rng
            sample = self.draw(n_sample)
        else:
            # multivariate rng
            sample = self.draw(n_sample)[:, which]
        lb, ub = truncate
        bins = np.arange(max(min(sample), lb), (
            min(max(sample), ub) + bin_width), bin_width)
        ax.hist(sample, bins, normed=1,
                alpha=0.3, color="grey", label=label)
        return ax, sample


class ProbIntegralGenerator(BaseRNG):
    def __init__(self, inv_cdf):
        self.inv_cdf = inv_cdf

    def draw(self, n=1):
        if n == 1:
            return self.inv_cdf(np.random.uniform())
        u = np.random.uniform(size=n)
        return np.array([self.inv_cdf(p) for p in u])


class RejectionGenerator(BaseRNG):
    def __init__(self, inducing_rng, rej_func):
        self.inducing_rng = inducing_rng
        self.rej_func = rej_func

    def draw(self, n=1):
        container, i = np.zeros(n), 0
        while i < n:
            y = self.inducing_rng.draw()
            u = np.random.uniform()
            if u <= self.rej_func(y):
                container[i] = y
                i += 1
        if n == 1:
            return container[0]
        return container


class GoldmanSachesMixture(BaseRNG):
    def draw(self, n=1):
        u = np.random.uniform(size=n)
        z = np.random.normal(size=n)
        z[u <= 0.82] *= 0.6
        z[u > 0.82] *= 1.98
        return z


def laplace_inv(b):
    return lambda u: np.random.choice([1, -1])*b*np.log(u)


def generalized_lambda_inv(l1, l2, l3, l4):
    return lambda u: l1 + (1/l2)*(u**l3 - (1-u)**l4)


def laplace_induced_normal_rej(c):
    return lambda x: (1/c)*np.sqrt(2*np.e/np.pi)*np.exp(
        -0.5*(x-np.sign(x))**2)


def weibull_inv(lamb, k):
    return lambda u: lamb*(-np.log(u))**(1/k)


def weibull_pdf(x_vec, lamb, k):
    return np.array([(k/lamb * (x/lamb) ** (k-1)) *
                     np.exp(-(x/lamb)**k) * (x >= 0) for x in x_vec])


def max_brownian_bdge_inv(b, h):
    return lambda u: (b+np.sqrt(b**2-2*np.log(u)*h))/2


def max_brownian_bdge_pdf(x_vec, b, h):
    return np.array([(x >= max(0,b))*np.exp(
        -2*x*(x-b)/h)*(4*x-2*b)/h for x in x_vec])


def cauchy_inv(x0, gamma):
    return lambda u: gamma*np.tan(np.pi*u-np.pi/2)+x0


def cauchy_pdf(x_vec, x0, gamma):
    return np.array([1/(
        np.pi*gamma*(1+((x-x0)/gamma)**2)) for x in x_vec])


def gumbel_inv(mu, beta):
    return lambda u: mu-beta*np.log(-np.log(u))


def gumbel_pdf(x_vec, mu, beta):
    return np.array([(1/beta)*np.exp(
        -(x-mu)/beta-np.exp(-(x-mu)/beta)) for x in x_vec])