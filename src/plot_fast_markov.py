#!/usr/bin/env python3

import datetime
import lzma
import multiprocessing
import pickle
import random
import sys

import boto3
import matplotlib.pyplot as plt
import numpy as np

if '../python' not in sys.path:
    sys.path.append('../python')
if './python' not in sys.path:
    sys.path.append('./python')

from uep_fast_run import *

def run_uep_map(params):
    res = simulation_results()
    run_uep(params, res)
    print(".", end="")
    return res

fixed_params = [{'avg_per': 1e-2,
                 'avg_bad_run': 100,
                 'overhead': 0.25},
                {'avg_per': 1e-2,
                 'avg_bad_run': 130,
                 'overhead': 0.25}]
ks = np.linspace(1000, 1400, 8, dtype=int).tolist()
k0_fraction = 0.1

base_params = simulation_params()
base_params.RFs[:] = [3, 1]
base_params.EF = 4
base_params.c = 0.1
base_params.delta = 0.5
base_params.L = 4
base_params.nblocks = 100

param_matrix = list()
for p in fixed_params:
    param_matrix.append(list())
    for k in ks:
        params = simulation_params(base_params)
        k0 = int(k0_fraction * k)
        params.Ks[:] = [k0, k - k0]
        params.chan_pGB = 1/p['avg_bad_run'] * p['avg_per'] / (1 - p['avg_per'])
        params.chan_pBG = 1/p['avg_bad_run']
        params.overhead = p['overhead']
        param_matrix[-1].append(params)

with multiprocessing.Pool() as pool:
    result_matrix = list()
    for ps in param_matrix:
        r = pool.map(run_uep_map, ps)
        result_matrix.append(r)
        print()

plt.figure()
plt.gca().set_yscale('log')

for (j, p) in enumerate(fixed_params):
    mib_pers = [ r.avg_pers[0] for r in result_matrix[j] ]
    lib_pers = [ r.avg_pers[1] for r in result_matrix[j] ]
    plt.plot(ks, mib_pers,
             marker='o',
             linewidth=1.5,
             label=("MIB E[#B] = {:.2f},"
                    " e = {:.0e},"
                    " t = {:.2f}".format(p['avg_bad_run'],
                                         p['avg_per'],
                                         p['overhead'])))
    plt.plot(ks, lib_pers,
             marker='o',
             linewidth=1.5,
             label=("LIB E[#B] = {:.2f},"
                    " e = {:.0e},"
                    " t = {:.2f}".format(p['avg_bad_run'],
                                         p['avg_per'],
                                         p['overhead'])))

plt.ylim(1e-8, 1)
plt.xlabel('K')
plt.ylabel('UEP PER')
plt.legend()
plt.grid()

s3 = boto3.resource('s3', region_name='us-east-1')
newid = random.getrandbits(64)
ts = int(datetime.datetime.now().timestamp())

plt.savefig('plot_fast_markov.pdf', format='pdf')
plotkey = "fast_plots/markov_{:d}_{:d}.pdf".format(ts, newid)
s3.meta.client.upload_file('plot_fast_markov.pdf',
                           'uep.zanol.eu',
                           plotkey,
                           ExtraArgs={'ACL': 'public-read'})
ploturl = ("https://s3.amazonaws.com/"
           "uep.zanol.eu/{!s}".format(plotkey))
print("Uploaded Markov plot at {!s}".format(ploturl))

plt.show()
