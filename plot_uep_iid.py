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
    def all(Ks, RFs, EF, c, delta, iid_per):
        return True

    @staticmethod
    def error_free(Ks, RFs, EF, c, delta, iid_per):
        return (
            iid_per == 0 and (
                (RFs == (1,1) and EF == 1) or
                (RFs == (5,1) and EF in [1, 2])
            )
        )

    @staticmethod
    def iid_errors(Ks, RFs, EF, c, delta, iid_per):
        return (
            any(math.isclose(iid_per, wp)
                for wp in [0.01, 0.1, 0.3]) and
            RFs == (5,1)
        )

    @staticmethod
    def only_eep(Ks, RFs, EF, c, delta, iid_per):
        return len(Ks) == 1 or all(rf == 1 for rf in RFs)

    @staticmethod
    def only_uep_zero(Ks, RFs, EF, c, delta, iid_per):
        return (len(Ks) > 1 and
                any(rf > 1 for rf in RFs) and
                iid_per == 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots the IID UEP results.',
                                     allow_abbrev=False)
    parser.add_argument("--param_filter", help="How to filter the data",
                        type=str, default="all")
    args = parser.parse_args()

    data = load_data_prefix("uep_iid_final/")

    git_sha1_set = sorted(set(d.get('git_sha1') or 'None' for d in data))
    print("Found {:d} commits:".format(len(git_sha1_set)))
    for s in git_sha1_set:
        print("  - " + s)

    # wanted_commits = [
    #     "57551110162e9e4f4d2094c1d86c6686a501fd96",
    # ]

    # data = [d for d in data if d.get('git_sha1') in wanted_commits]

    print("Using {:d} data packs".format(len(data)))

    param_set = sorted(set((tuple(d['Ks']),
                            tuple(d['RFs']),
                            d['EF'],
                            d['c'],
                            d['delta'],
                            d['iid_per']) for d in data))

    # Old results no not have the avg_ripples
    # for d in data:
    #     if 'avg_ripples' not in d:
    #         d['avg_ripples'] = [float('nan') for o in d['overheads']]

    param_filter = getattr(param_filters, args.param_filter)
    assert(callable(param_filter))

    p = plots()
    p.automaticXScale = True
    #p.automaticXScale = [0,0.3]
    p.automaticYScale = [1e-8, 1]
    p.add_plot(plot_name='per',xlabel='Overhead',ylabel='PER',logy=True)
    p.add_plot(plot_name='nblocks',xlabel='Overhead',ylabel='nblocks',logy=False)
    #p.add_plot(plot_name='ripple',xlabel='Overhead',ylabel='ripple',logy=False)
    p.add_plot(plot_name='drop_rate',xlabel='Overhead',ylabel='drop_rate',logy=False)

    for params in param_set:
        data_same_pars = [d for d in data if (tuple(d['Ks']),
                                              tuple(d['RFs']),
                                              d['EF'],
                                              d['c'],
                                              d['delta'],
                                              d['iid_per']) == params]
        Ks = params[0]
        RFs = params[1]
        EF = params[2]
        c = params[3]
        delta = params[4]
        iid_per = params[5]

        overheads = sorted(set(o for d in data_same_pars for o in d['overheads']))

        # overheads = sorted(set(overheads).intersection(np.linspace(0, 0.4, 16)))

        avg_pers = np.zeros((len(overheads), len(Ks)))
        nblocks = np.zeros(len(overheads), dtype=int)
        avg_ripples = np.zeros(len(overheads))
        avg_drop_rates = np.zeros(len(overheads))
        for i, oh in enumerate(overheads):
            avg_counters = [AverageCounter() for k in Ks]
            avg_ripple = AverageCounter()
            avg_drop = AverageCounter()
            for d in data_same_pars:
                for l, d_oh in enumerate(d['overheads']):
                    if d_oh != oh: continue

                    for j,k in enumerate(Ks):
                        per = d['error_counts'][l][j] / (k * d['nblocks'])
                        avg_counters[j].add(per, d['nblocks'])
                    if not math.isnan(d['avg_ripples'][l]):
                        avg_ripple.add(d['avg_ripples'][l],
                                       d['nblocks'])
                    avg_drop.add(d['avg_drops'][l], d['nblocks'])

            avg_ripples[i] = avg_ripple.avg
            avg_drop_rates[i] = avg_drop.avg
            avg_pers[i,:] = [c.avg for c in avg_counters]
            nblocks[i] = avg_counters[0].total_weigth

        if not param_filter(*params): continue

        # Average into a single PER when EEP
        if len(RFs) > 1 and all(rf == 1 for rf in RFs):
            new_pers = np.zeros((avg_pers.shape[0], 1))
            for i, ps in enumerate(avg_pers):
                avg_p = sum(p*k for p,k in zip(ps, Ks)) / sum(Ks)
                new_pers[i] = avg_p
            avg_pers = new_pers

        legend_str = ("Ks={!s},"
                      "RFs={!s},"
                      "EF={:d},"
                      "c={:.2f},"
                      "delta={:.2f},"
                      "e={:.0e}").format(*params)

        typestr = 'mib'
        if all(rf == 1 for rf in RFs) or len(Ks) == 1:
            typestr = 'eep'

        mibline = p.add_data(plot_name='per',label=legend_str,type=typestr,
                           x=overheads, y=avg_pers[:,0])
        if len(Ks) > 1 and any(rf != 1 for rf in RFs):
            p.add_data(plot_name='per',label=legend_str,type='lib',
                       x=overheads, y=avg_pers[:,1],
                       color=mibline.get_color())
        plt.grid()

        p.add_data(plot_name='nblocks',label=legend_str,
                   x=overheads, y=nblocks)
        plt.autoscale(enable=True, axis='y', tight=False)

        # p.add_data(plot_name='ripple',label=legend_str,
        #            x=overheads, y=avg_ripples)
        # plt.autoscale(enable=True, axis='y', tight=False)

        p.add_data(plot_name='drop_rate',label=legend_str,
                   x=overheads, y=avg_drop_rates)
        plt.autoscale(enable=True, axis='y', tight=False)

        #the_oh_is = [i for i,oh in enumerate(overheads)
        #             if math.isclose(oh, 0.24)]
        #if len(the_oh_is) > 0:
        #    the_oh_i = the_oh_is[0]
        #    print("At overhead {:.2f}:".format(overheads[the_oh_i]))
        #    print(" " * 4 + legend_str, end="")
        #    print(" -> MIB={:e}, LIB={:e}".format(avg_pers[the_oh_i, 0],
        #                                          avg_pers[the_oh_i, 1]))

    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_plot_pdf(p.get_plot('per'),'iid/{}/{} {}'.format(args.param_filter,
                                                          p.describe_plot('per'),
                                                          datestr))
    save_plot_pdf(p.get_plot('nblocks'),'iid/{}/{} {}'.format(args.param_filter,
                                                              p.describe_plot('nblocks'),
                                                              datestr))
    save_plot_pdf(p.get_plot('drop_rate'),'iid/{}/{} {}'.format(args.param_filter,
                                                                p.describe_plot('drop_rate'),
                                                                datestr))

    plt.show()
