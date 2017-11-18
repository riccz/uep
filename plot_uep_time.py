import argparse
import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np

from plots import *
from uep import *
from utils.aws import *
from utils.plots import *
from utils.stats import *

class param_filters:
    @staticmethod
    def all(Ks, RFs, EF, c, delta):
        return True

    @staticmethod
    def eep(Ks, RFs, EF, c, delta):
        return (len(Ks) == 1 or
                all(rf == 1 for rf in RFs))

    @staticmethod
    def uep(*args):
        return not param_filters.eep(*args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots the UEP timing results.',
                                     allow_abbrev=False)
    parser.add_argument("--param_filter", help="How to filter the data",
                        type=str, default="all")
    args = parser.parse_args()

    data = load_data_prefix("uep_time/")

    git_sha1_set = sorted(set(d.get('git_sha1') or 'None' for d in data))
    print("Found {:d} commits:".format(len(git_sha1_set)))
    for s in git_sha1_set:
        print("  - " + s)

    wanted_commits = [
    ]

    if len(wanted_commits) > 0:
        data = list(filter(lambda d: d.get('git_sha1') in wanted_commits, data))

    print("Using {:d} data packs".format(len(data)))

    param_set = sorted(set((tuple(d['Ks']),
                            tuple(d['RFs']),
                            d['EF'],
                            d['c'],
                            d['delta']) for d in data))

    param_filter = getattr(param_filters, args.param_filter)
    assert(callable(param_filter))

    p = plots()
    p.automaticXScale = [0, 0.8]

    p.add_plot(plot_name='enc_time',xlabel='Overhead',
               ylabel='Encoding time [ms]',logy=False)
    p.add_plot(plot_name='dec_time',xlabel='Overhead',
               ylabel='Decoding time [ms]',logy=False)
    p.add_plot(plot_name='nblocks',xlabel='Overhead',ylabel='nblocks',logy=False)

    for params in param_set:
        data_same_pars = [d for d in data if (tuple(d['Ks']),
                                              tuple(d['RFs']),
                                              d['EF'],
                                              d['c'],
                                              d['delta']) == params]
        if len(data_same_pars) > 1:
            print("Cannot average the execution times")
            sys.exit(1)

        Ks = params[0]
        RFs = params[1]
        EF = params[2]
        c = params[3]
        delta = params[4]

        overheads = sorted(set(o for d in data_same_pars for o in d['overheads']))

        # overheads = sorted(set(overheads).intersection(np.linspace(0, 0.4, 16)))

        avg_enc_times = np.zeros(len(overheads))
        avg_dec_times = np.zeros(len(overheads))
        nblocks = np.zeros(len(overheads), dtype=int)
        for i, oh in enumerate(overheads):
            avg_encs = AverageCounter()
            avg_decs = AverageCounter()
            for d in data_same_pars:
                for l, d_oh in enumerate(d['overheads']):
                    if d_oh != oh: continue

                    avg_encs.add(d['avg_enc_times'][l], d['nblocks'])
                    avg_decs.add(d['avg_dec_times'][l], d['nblocks'])

            avg_enc_times[i] = avg_encs.avg * 1000 # ms
            avg_dec_times[i] = avg_decs.avg * 1000 # ms
            nblocks[i] = avg_encs.total_weigth

        if not param_filter(*params): continue

        # legend_str = ("Ks={!s},"
        #               "RFs={!s},"
        #               "EF={:d},"
        #               "c={:.2f},"
        #               "delta={:.2f}").format(*params)
        legend_str = "EF = {:d}".format(EF)

        p.automaticYScale = [50, 130]
        p.add_data(plot_name='enc_time',label=legend_str,
                           x=overheads, y=avg_enc_times)

        p.automaticYScale = [0, 8]
        p.add_data(plot_name='dec_time',label=legend_str,
                           x=overheads, y=avg_dec_times)

        p.add_data(plot_name='nblocks',label=legend_str,
                   x=overheads, y=nblocks)
        plt.autoscale(enable=True, axis='y', tight=False)

    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_plot_pdf(p.get_plot('enc_time'),'time/{}/{} {}'.format(args.param_filter,
                                                                p.describe_plot('enc_time'),
                                                                datestr))
    save_plot_pdf(p.get_plot('dec_time'),'time/{}/{} {}'.format(args.param_filter,
                                                                p.describe_plot('dec_time'),
                                                                datestr))
    save_plot_pdf(p.get_plot('nblocks'),'time/{}/{} {}'.format(args.param_filter,
                                                               p.describe_plot('nblocks'),
                                                               datestr))

    plt.show()
