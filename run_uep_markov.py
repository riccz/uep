import argparse
import datetime
import lzma
import math
import pickle
import random
import subprocess

import numpy as np

from uep import *
from utils.aws import *
from utils.stats import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs a Markov UEP simulation.')
    parser.add_argument("rf", help="MIB Repeating factor",type=int)
    parser.add_argument("ef", help="Expanding factor",type=int)
    parser.add_argument("nblocks", help="nblocks for the simulation",type=int)
    parser.add_argument("piB", help="1 over the steady state prob. of being in a Bad slot.",type=float)
    parser.add_argument("EnB", help="Avg. number of contiguous Bad slots",type=int)
    parser.add_argument("--overhead", help="Overhead",type=float, default=0.25)
    parser.add_argument("--k_min", help="Take only K >= this",
                        type=float,
                        default=float('-inf'))
    parser.add_argument("--k_max", help="Take only K <= this",
                        type=float,
                        default=float('inf'))
    args = parser.parse_args()

    git_sha1 = None
    try:
        git_proc = subprocess.run(["git", "log",
                                   "-1",
                                   "--format=%H"],
                                  check=False,
                                  stdout=subprocess.PIPE)
        if git_proc.returncode == 0:
            git_sha1 = git_proc.stdout.strip().decode()
    except FileNotFoundError:
        pass

    if git_sha1 is None:
        try:
            git_sha1 = open('git_commit_sha1', 'r').read().strip()
        except FileNotFound:
            pass

    RFs = [args.rf, 1]
    EF = args.ef

    c = 0.1
    delta = 0.5

    overhead = args.overhead

    avg_per = 1 / args.piB
    avg_bad_run = args.EnB

    nblocks = args.nblocks

    Ks_frac = [0.05, 0.95]
    k_blocks = np.linspace(100, 15100, 16, dtype=int).tolist()
    k_blocks += np.linspace(100, 2100, 11, dtype=int)[:-1].tolist()
    k_blocks = sorted(set(k_blocks))

    k_blocks = [filter(lambda k: (k >= args.k_min and
                                  k <= args.k_max),
                       k_blocks)]

    used_rngstate = random.getstate()

    pBG = 1/avg_bad_run
    assert(pBG >= 0 and pBG <= 1)
    pGB = pBG * avg_per / (1 - avg_per)
    assert(pGB >= 0 and pGB <= 1)

    avg_pers = np.zeros((len(k_blocks), len(Ks_frac)))
    avg_drops = np.zeros(len(k_blocks))
    used_Ks = np.zeros((len(k_blocks), len(Ks_frac)), dtype=int)
    error_counts = np.zeros((len(k_blocks), len(Ks_frac)), dtype=int)
    for j, k_block in enumerate(k_blocks):
        Ks = [int(round(k_block * kf)) for kf in Ks_frac]
        if sum(Ks) < k_block:
            while k_block - sum(Ks) > 0:
               Ks[np.argmin(Ks)] += 1
        elif sum(Ks) > k_block:
            while sum(Ks) - k_block > 0:
               Ks[np.argmax(Ks)] -= 1
        assert(sum(Ks) == k_block)
        assert(all(ki > 0 for ki in Ks))
        used_Ks[j] = Ks

        print("Run with Ks = {!s}".format(Ks))

        sim = UEPSimulation(Ks=Ks, RFs=RFs, EF=EF, c=c, delta=delta,
                            overhead=overhead,
                            markov_pGB=pGB,
                            markov_pBG=pBG,
                            nblocks=nblocks)
        results = run_parallel(sim)
        avg_pers[j] = results['error_rates']
        avg_drops[j] = results['drop_rate']
        error_counts[j] = results['error_counts']
        print("  PERs = {!s}".format(avg_pers[j]))
        print("  errors = {!s}".format(error_counts[j]))

    newid = random.getrandbits(64)
    save_data("uep_markov_final/uep_vs_k_markov_{:d}.pickle.xz".format(newid),
              git_sha1=git_sha1,
              timestamp=datetime.datetime.now().timestamp(),
              used_rngstate=used_rngstate,
              k_blocks=k_blocks,
              Ks_frac=Ks_frac,
              used_Ks=used_Ks,
              RFs=RFs,
              EF=EF,
              c=c,
              delta=delta,
              overhead=overhead,
              avg_per=avg_per,
              avg_bad_run=avg_bad_run,
              pBG=pBG,
              pGB=pGB,
              nblocks=nblocks,
              avg_pers=avg_pers,
              avg_drops=avg_drops,
              error_counts=error_counts)
