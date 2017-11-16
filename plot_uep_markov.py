import argparse
import datetime
import sys

import boto3
import matplotlib.pyplot as plt
import numpy as np

from plots import *
from uep import *
from utils.aws import *
from utils.plots import *
from utils.stats import *

class param_filters:
    @staticmethod
    def all(RFs, EF, c, delta, overhead, avg_per, avg_bad_run, Ks_frac):
        return True

    @staticmethod
    def error_free(RFs, EF, c, delta, overhead, avg_per, avg_bad_run, Ks_frac):
        return (
            avg_per == 0 and avg_bad_run == 1 and (
                RFs in [(1,1), (5,1)]
            )
        )

    @staticmethod
    def br(wanted_br, RFs, EF, c, delta, overhead, avg_per, avg_bad_run, Ks_frac):
        return (
            avg_bad_run == wanted_br and
            RFs in [(1,1), (5,1)]
        )

    @staticmethod
    def br5(*args):
        return param_filters.br(5, *args)

    @staticmethod
    def br10(*args):
        return param_filters.br(10, *args)

    @staticmethod
    def br50(*args):
        return param_filters.br(50, *args)

    @staticmethod
    def pi100(RFs, EF, c, delta, overhead, avg_per, avg_bad_run, Ks_frac):
        return (
            avg_per != 0 and
            math.isclose(1/avg_per, 100) and
            RFs in [(1,1), (5,1)]
        )

    @staticmethod
    def pi10(RFs, EF, c, delta, overhead, avg_per, avg_bad_run, Ks_frac):
        return (
            avg_per != 0 and
            math.isclose(1/avg_per, 10) and
            RFs in [(1,1), (5,1)]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots the Markov UEP results.',
                                     allow_abbrev=False)
    parser.add_argument("--param_filter", help="How to filter the data",
                        type=str, default="all")
    parser.add_argument("--merge", action='store_true',
                        help="Merge the data packs")
    args = parser.parse_args()

    data = load_data_prefix_2("uep_markov_final/")

    git_sha1_set = sorted(set(d[0].get('git_sha1') or 'None' for d in data))
    print("Found {:d} commits:".format(len(git_sha1_set)))
    for s in git_sha1_set:
        print("  - " + s)

    # wanted_commits = [
    #     "921f5e0f5bf82591ec93f215cd490f8bdd9473fe",
    #     "a616ad6478ac1f80dc3f7983cbf5b4958bf9fcfb",
    # ]

    # data = [d for d in data if d[0].get('git_sha1') in wanted_commits]

    print("Using {:d} data packs".format(len(data)))

    if args.merge:
        print("WARNING: The data packs will be merged by"
              " averaging. The commit, timestamp and rng"
              " seed will be lost.")
        yn = input("Proceed? yes/[no]: ")
        if yn.strip() not in ['yes', 'y']:
            sys.exit(1)

    param_set = sorted(set((tuple(d[0]['RFs']),
                            d[0]['EF'],
                            d[0]['c'],
                            d[0]['delta'],
                            d[0]['overhead'],
                            d[0]['avg_per'],
                            d[0]['avg_bad_run'],
                            tuple(d[0]['Ks_frac'])) for d in data))

    param_filter = getattr(param_filters, args.param_filter)
    assert(callable(param_filter))

    p = plots()
    p.automaticXScale = True
    #p.automaticXScale = [0,0.3]
    p.automaticYScale = [1e-8, 1]
    p.add_plot(plot_name='per',xlabel='K',ylabel='PER',logy=True)
    p.add_plot(plot_name='nblocks',xlabel='K',ylabel='nblocks',logy=False)
    p.add_plot(plot_name='drop_rate',xlabel='K',ylabel='drop_rate',logy=False)

    merge_output = list()
    for params in param_set:
        data_same = [d for d in data if (tuple(d[0]['RFs']),
                                         d[0]['EF'],
                                         d[0]['c'],
                                         d[0]['delta'],
                                         d[0]['overhead'],
                                         d[0]['avg_per'],
                                         d[0]['avg_bad_run'],
                                         tuple(d[0]['Ks_frac'])) == params]
        data_same_pars = [d[0] for d in data_same]
        data_same_keys = [d[1] for d in data_same]

        RFs = params[0]
        EF = params[1]
        c = params[2]
        delta = params[3]
        overhead = params[4]
        avg_per = params[5]
        avg_bad_run = params[6]
        Ks_frac = params[7]
        pGB = data_same_pars[0]['pGB']
        pBG = data_same_pars[0]['pBG']

        k_blocks = sorted(set(k for d in data_same_pars for k in d['k_blocks']))

        avg_pers = np.zeros((len(k_blocks), len(Ks_frac)))
        nblocks = np.zeros(len(k_blocks), dtype=int)
        avg_drop_rates = np.zeros(len(k_blocks))
        used_Ks = np.zeros((len(k_blocks), len(Ks_frac)), dtype=int)
        error_counts = np.zeros((len(k_blocks), len(Ks_frac)), dtype=int)
        for i, k_block in enumerate(k_blocks):
            avg_counters = [AverageCounter() for k in Ks_frac]
            avg_drop = AverageCounter()
            for d in data_same_pars:
                for l, d_k_block in enumerate(d['k_blocks']):
                    if d_k_block != k_block: continue

                    if hasattr(d['nblocks'], '__len__'):
                        d_nblocks = d['nblocks'][l]
                    else:
                        d_nblocks = d['nblocks']

                    for j, _ in enumerate(Ks_frac):
                        errc = d['error_counts'][l][j]
                        error_counts[i, j] += errc
                        per = errc / (d['used_Ks'][l][j] * d_nblocks)
                        avg_counters[j].add(per, d_nblocks)
                    avg_drop.add(d['avg_drops'][l], d_nblocks)
                    used_Ks[i] = d['used_Ks'][l]
            avg_pers[i,:] = [c.avg for c in avg_counters]
            nblocks[i] = avg_counters[0].total_weigth
            avg_drop_rates[i] = avg_drop.avg

        if args.merge:
            merged = {
                'timestamp': datetime.datetime.now().timestamp(),
                'merged': True,
                'k_blocks': k_blocks,
                'Ks_frac': Ks_frac,
                'RFs': RFs,
                'EF': EF,
                'c': c,
                'delta': delta,
                'overhead': overhead,
                'avg_per': avg_per,
                'avg_bad_run': avg_bad_run,
                'pGB': pGB,
                'pBG': pBG,
                'nblocks': nblocks,
                'avg_pers': avg_pers,
                'error_counts': error_counts,
                'avg_drops': avg_drop_rates,
            }
            merge_output.append((merged, data_same_keys))

        if not param_filter(*params): continue

        # Average into a single PER when EEP
        if len(RFs) > 1 and all(rf == 1 for rf in RFs):
            new_pers = np.zeros((avg_pers.shape[0], 1))
            for i, ps in enumerate(avg_pers):
                avg_p = (sum(p*k for p,k in zip(ps, used_Ks[i])) /
                         sum(used_Ks[i]))
                new_pers[i] = avg_p
            avg_pers = new_pers

        legend_str = ("RFs={!s},"
                      "EF={:d},"
                      "c={:.2f},"
                      "delta={:.2f},"
                      "oh={:.2f},"
                      "avg_per={:.0e},"
                      "avg_bad_run={:.2f},"
                      "Ks_frac={!s}").format(*params)

        typestr = 'mib'
        if all(rf == 1 for rf in RFs) or len(Ks_frac) == 1:
            typestr = 'eep'

        mibline = p.add_data(plot_name='per',label=legend_str,type=typestr,
                           x=k_blocks, y=avg_pers[:,0])
        if len(Ks_frac) > 1 and any(rf != 1 for rf in RFs):
            p.add_data(plot_name='per',label=legend_str,type='lib',
                       x=k_blocks, y=avg_pers[:,1],
                       color=mibline.get_color())
        plt.grid()

        p.add_data(plot_name='nblocks',label=legend_str,
                   x=k_blocks, y=nblocks)
        plt.autoscale(enable=True, axis='y', tight=False)

        p.add_data(plot_name='drop_rate',label=legend_str,
                   x=k_blocks, y=avg_drop_rates)
        plt.autoscale(enable=True, axis='y', tight=False)

        #the_oh_is = [i for i,oh in enumerate(overheads)
        #             if math.isclose(oh, 0.24)]
        #if len(the_oh_is) > 0:
        #    the_oh_i = the_oh_is[0]
        #    print("At overhead {:.2f}:".format(overheads[the_oh_i]))
        #    print(" " * 4 + legend_str, end="")
        #    print(" -> MIB={:e}, LIB={:e}".format(avg_pers[the_oh_i, 0],
        #                                          avg_pers[the_oh_i, 1]))

    for m in merge_output:
        newid = random.getrandbits(64)
        newkey = "uep_markov_final/uep_vs_k_markov_{:d}_merged.pickle.xz".format(newid)
        save_data(newkey, **(m[0]))
        print("Uploaded merged data pack: " + newkey)
        bucket = boto3.resource('s3').Bucket('uep.zanol.eu')
        for k in m[1]:
            bucket.copy({'Bucket': 'uep.zanol.eu', 'Key': k},
                        "backup/"+k,
                        ExtraArgs={'ACL': 'public-read'})
        print("Backed up old packs")
        bucket.delete_objects(Delete={'Objects':[{'Key':k} for k in m[1]]})
        print("Deleted old packs")


    datestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_plot_pdf(p.get_plot('per'),'markov/{}/{} {}'.format(args.param_filter,
                                                             p.describe_plot('per'),
                                                             datestr))
    save_plot_pdf(p.get_plot('nblocks'),'markov/{}/{} {}'.format(args.param_filter,
                                                                 p.describe_plot('nblocks'),
                                                                 datestr))
    save_plot_pdf(p.get_plot('drop_rate'),'markov/{}/{} {}'.format(args.param_filter,
                                                                   p.describe_plot('drop_rate'),
                                                                   datestr))

    plt.show()
