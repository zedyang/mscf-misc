import inspect
import random
from copy import copy
from functools import singledispatch

import numpy as np
import pandas as pd
from numpy.linalg import inv
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin

from lazy import lazy
from tabulate import tabulate


class LinearModel(RegressorMixin, BaseEstimator, TransformerMixin):

    def __init__(self):
        super(LinearModel, self).__init__()
        self.x_names, self.y_name = None, None
        self.n, self.p = None, None
        self.formula = 'LinearModel( )\n'
        self.fitted = False

        # algebraic properties
        self.X, self.y = None, None
        self.XtX = None
        self.XtX_inv = None
        self.b_hat = None
        self.HX = None
        self.residuals = None  # n*1
        self.RSS = None

        # statistical properties
        self.sigma_sq_mle = None
        self.sigma_sq_unbiased = None
        self.rse = None
        self.cov_b = None
        self.var_b = None
        self.std_b = None
        self.t_stats = None
        self.p_val = None

    def __repr__(self):
        _repr, rows = self.formula, []
        if not self.fitted:
            return _repr
        for i, s in enumerate(self.x_names):
            rows.append([
                s, self.b_hat[i], self.std_b[i],
                self.t_stats[i], self.p_val[i]])
        _repr += tabulate(
            rows, headers=['', 'Estimate', 'Std.Error',
                           't.Stat', 'p.value'])
        return _repr

    def fit_df(self, data, formula):
        """
        :param data:
        :param formula:
        :return:
        """
        self.formula = 'LinearModel( {} )\n\n'.format(formula)
        y_var, x_var = formula.split('~')
        x_vars = [x.strip() for x in x_var.split('+')]
        self.x_names = [x for x in x_vars if x != '0']
        self.y_name = y_var
        self.n, self.p = len(data), len(x_vars) - 1
        X = data[self.x_names].values
        if '0' not in x_vars:
            self.p += 2
            self.x_names = ['1'] + self.x_names
            X = np.concatenate((np.ones((self.n, 1)), X), axis=1)
        y = data[[y_var.strip()]].values

        return self.fit(X, y)

    def fit(self, X, y):
        """
        Fit the model.
        :param X: features set.
        :param y: response variable set.
        :return:
        """
        n, p = X.shape
        # algebraic properties
        self.X, self.y = X, y
        self.XtX = X.T.dot(X)
        self.XtX_inv = inv(self.XtX)
        self.b_hat = self.XtX_inv.dot(X.T.dot(y))
        self.HX = X.dot(self.XtX_inv).dot(X.T)
        self.residuals = y - X.dot(self.b_hat)  # n*1
        self.RSS = sum([r[0] * r[0] for r in self.residuals])

        # statistical properties
        self.sigma_sq_mle = self.RSS / n
        self.sigma_sq_unbiased = self.RSS / (n - p)
        self.rse = np.sqrt(self.sigma_sq_unbiased)
        self.cov_b = self.XtX_inv * self.sigma_sq_unbiased
        self.var_b = np.diag(self.cov_b).reshape(p, 1)
        self.std_b = np.sqrt(self.var_b).reshape(p, 1)
        self.t_stats = self.b_hat / self.std_b
        self.p_val = (stats.t.sf(
            np.abs(self.t_stats), n - p) * 2).reshape(p, 1)

        self.fitted = True
        return self

    def predict(self, X_test):
        return X_test.dot(self.b_hat)

    @property
    def coef(self):
        return np.append(self.b_hat, np.array([
            self.sigma_sq_unbiased]).reshape(1, 1), axis=0)

    def full_cov(self, cov):
        """
        covariance matrix with that of sigma_sq.
        :param cov:
        :return:
        """
        full = np.concatenate(
            (cov, np.array([[0.0] * self.p])), axis=0)
        col = np.array([[0.0] * (self.p + 1)])
        col[0][-1] = 2 * self.sigma_sq_unbiased ** 2
        return np.concatenate(
            (full, col.T), axis=1)

    def hccme(self, f_inflate):
        """
        "Sandwich" estimator of covariance matrix,
        different variance-inflation approaches are used.
         - hc0: hat{epsilon}_i = hat{u}_i
         - hc1: hat{epsilon}_i = hat{u}_i * (n/df)
         - hc2: hat{epsilon}_i = hat{u}_i * (1/(1-h_i))
         - hc3: hat{epsilon}_i = hat{u}_i * (1/(1-h_i))^2
        :param f_inflate:
        :return:
        """
        Omega = np.diag([
            r[0] * r[0] * f_inflate(i)
            for i, r in enumerate(self.residuals)])
        XtOmegaX = self.X.T.dot(Omega).dot(self.X)
        return self.XtX_inv.dot(XtOmegaX).dot(self.XtX_inv)

    @lazy
    def hc0(self):
        inflate_0 = lambda i: 1
        return self.hccme(inflate_0)

    @lazy
    def hc1(self):
        inflate_1 = lambda i: self.n / (self.n - self.p)
        return self.hccme(inflate_1)

    @lazy
    def hc2(self):
        inflate_2 = lambda i: 1 / (1 - self.HX[i][i])
        return self.hccme(inflate_2)

    @lazy
    def hc3(self):
        """
        The jackknife approach
        :return:
        """
        inflate_3 = lambda i: 1 / (1 - self.HX[i][i]) ** 2
        return self.hccme(inflate_3)

    @staticmethod
    def auto_gradient(f):
        # return the gradient of f where
        # f = f(xy=(X,y), arg0, arg1, ...)
        n_args = len(inspect.getfullargspec(f).args) - 1

        def grad(xy, pos, delta=1e-6):
            g = [0] * n_args
            for i, _ in enumerate(g):
                args_left = copy(pos)
                args_left[i] -= delta
                f_left = f(xy, *args_left)
                args_right = copy(pos)
                args_right[i] += delta
                f_right = f(xy, *args_right)
                g[i] = (f_right - f_left) / (2 * delta)
            return np.array(g).reshape(len(pos), 1)

        return grad

    def __boot_sample(self):
        # stat: stat(X, coefs)
        b = random.choices(range(self.n), k=self.n)
        data = (self.X[b, ], self.y[b, ])
        return data

    def np_boot(self, stat, b=1000, verbose=1000):
        stat_values = []
        for i in range(b):
            Xb, yb = self.__boot_sample()
            lm = LinearModel()
            lm.fit(Xb, yb)
            coef_b = lm.coef
            stat_b = stat(
                (Xb, yb),
                *[x[0] for x in coef_b])
            stat_values.append(stat_b)
            if not i % verbose:
                print("Finished {}/{} boot samples".format(i, b))
        return stat_values

    def param_boot(self, stat, b=1000, verbose=1000):
        stat_values = []
        for i in range(b):
            Xb, _ = self.__boot_sample()
            y_hat = self.predict(Xb)
            yb = y_hat + np.random.normal(
                loc=0.0, scale=self.rse,
                size=self.n).reshape(self.n, 1)
            lm = LinearModel()
            lm.fit(Xb, yb)
            coef_b = lm.coef
            stat_b = stat(
                (Xb, yb),
                *[x[0] for x in coef_b])
            stat_values.append(stat_b)
            if not i % verbose:
                print("Finished {}/{} boot samples".format(i, b))
        return stat_values