import numpy as np
from copy import copy
from scipy.optimize import newton
from scipy.interpolate import interp1d


class Curve(object):
    def __init__(self, formula):
        self.__call = formula

    def __call__(self, *args, **kwargs):
        return self.__call(*args, **kwargs)

    @staticmethod
    def calibrate(cashflow, price, n=2):
        n = newton(func=lambda y: cashflow.price(
            Curve.SingleRateCurve(y, n))-price,
               x0=0.001, tol=1.48e-9, maxiter=80)
        return n

    @classmethod
    def SingleRateCurve(cls, y, n=2):
        def d(t):
            return 1/(1+y/n)**(n*t)
        return Curve(d)

    @classmethod
    def SpotRateDict(cls, spots, n=2):
        disc = dict()
        for t in spots:
            disc[t] = 1/(1+spots[t]/n)**(n*t)
        return disc

    @classmethod
    def LinearInterpolation(cls, anchors):
        profile = [(t, r) for t, r in anchors]
        profile = sorted(profile, key=lambda a: a[0])
        t_vec = map(lambda a: a[0], profile)
        r_vec = map(lambda a: a[1], profile)
        d = interp1d(t_vec, r_vec)
        return Curve(d)

