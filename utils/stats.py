import concurrent.futures as cf
import datetime
import lzma
import math
import multiprocessing
import os
import os.path
import pickle
import random
import re
import sys

import boto3
import botocore

from dateutil.tz import tzutc


def phi(n, z, alpha):
    from scipy import stats

    assert(n >= 2)
    assert(z >= 0 and z <= n)
    assert(alpha >= 0 and alpha <= 1)
    n1 = 2*(z+1)
    n2 = 2*(n-z)
    f = stats.f.ppf(1-alpha, n1, n2)
    return n1*f / (n2 + n1*f)

def bernoulli_ci(z, n, gamma):
    def L(z):
        if z == 0:
            return 0
        else:
            return phi(n, z-1, (1+gamma) / 2)

    l = L(z)
    u = 1 - L(n-z)
    return (l, u)

def success_err(z, n):
    (l,u) = bernoulli_ci(z, n, 0.95)
    return (-(l - z/n), u - z/n)

class AverageCounter:
    def __init__(self, avg = 0, sum_w = 0):
        self.__sum_w = sum_w
        self.__avg = avg

    def add(self, x, w):
        new_sumw = self.__sum_w + w
        self.__avg = (self.__sum_w * self.__avg +
                      x * w) / new_sumw
        self.__sum_w = new_sumw

    @property
    def avg(self):
        return self.__avg

    @property
    def total_weigth(self):
        return self.__sum_w

def update_average(m, sumw, s, w):
    c = AverageCounter(m, sumw)
    c.add(s, w)
    return c.avg

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)
