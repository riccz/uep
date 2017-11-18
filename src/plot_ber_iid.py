#!/usr/bin/env python3

import datetime
import decimal
import lzma
import numbers
import operator
import os
import pickle
import random
import shelve
import subprocess
import sys

import boto3
import matplotlib.pyplot as plt
import numpy as np
import botocore

from boto3.dynamodb.conditions import Attr, Key
from decimal import Decimal
from operator import itemgetter
from subprocess import check_call

from ber import *
from utils.aws import *
from utils.logs import *
from utils.stats import *

if __name__ == "__main__":
    config = botocore.client.Config(read_timeout=300)
    s3 = boto3.resource('s3', config=config, region_name='us-east-1')
    uep_bucket = s3.Bucket('uep.zanol.eu')
    dyn = boto3.resource('dynamodb', region_name='us-east-1')
    sim_tab = dyn.Table('uep_sim_results')

    wanted_pers = [0.01, 0.1, 0.3]
    # Get all the iid points with given params, partition by iid_per
    udp_err_rates = []
    points = []
    for wp in wanted_pers:
        dec_wp = Decimal(wp).quantize(Decimal('1e-8'))
        q = sim_tab.query(IndexName='iid_per-overhead-index',
                          KeyConditionExpression=Key('iid_per').eq(dec_wp),
                          FilterExpression=Attr('k0').eq(100) &
                          Attr('k1').eq(1900) &
                          Attr('rf0').eq(5) &
                          Attr('rf1').eq(1) &
                          Attr('ef').eq(2))# &
                          #Attr('stream_name').eq('stefan_cif_long'))# &
                          #Attr('masked').ne(True))
        for p in q['Items']:
            per = p['iid_per']
            if per not in udp_err_rates:
                udp_err_rates.append(per)
                points.append([])

            i = udp_err_rates.index(per)
            points[i].append(p)

    s = sorted(zip(udp_err_rates, points), key=itemgetter(0))
    udp_err_rates = [ e[0] for e in s ]
    points = [ e[1] for e in s]

    overheads = [ [] for p in points ]
    udp_real_err_rates = [ [] for p in points ] # Measured UDP err rates
    uep_err_rates = [ [] for p in points ] # Measured UEP err rates
    uep_err_counts = [ [] for p in points ] # Cumulative err count for each sub-block
    uep_tot_pkts = [ [] for p in points ]

    for (j, per) in enumerate(udp_err_rates):
        for p in points[j]:
            bs_o = uep_bucket.Object(p['s3_ber_scanner']).get()
            bs = pickle.loads(lzma.decompress(bs_o['Body'].read()))
            print("[{!s}] iid_per = {:f}, overhead = {:f}"
                  " results:".format(p['result_id'],
                                     p['iid_per'],
                                     p['overhead']))

            errs = list(map(operator.sub,
                            bs.tot_sent,
                            bs.tot_recvd))

            print("UDP err rate = {:e}".format(bs.udp_err_rate))
            print("UEP err rates = {!s}".format(bs.uep_err_rates))
            print("UEP err counts = {!s}".format(errs))
            print("UEP pkt counts = {!s}".format(bs.tot_sent))

            if p['overhead'] not in overheads[j]: # First point at this overhead
                overheads[j].append(p['overhead'])
                udp_real_err_rates[j].append(bs.udp_err_rate)
                uep_err_rates[j].append(bs.uep_err_rates)
                uep_err_counts[j].append(errs)
                uep_tot_pkts[j].append(bs.tot_sent)
            else:
                i = overheads[j].index(p['overhead'])
                sumw = sum(uep_tot_pkts[j][i])
                w = sum(bs.tot_sent)
                udp_real_err_rates[j][i] = update_average(udp_real_err_rates[j][i],
                                                          sumw,
                                                          bs.udp_err_rate,
                                                          w)
                uep_err_rates[j][i][:] = map(lambda m, s: update_average(m, sumw, s, w),
                                             uep_err_rates[j][i],
                                             bs.uep_err_rates)
                uep_err_counts[j][i][:] = map(operator.add,
                                              uep_err_counts[j][i],
                                              errs)
                uep_tot_pkts[j][i][:] = map(operator.add,
                                            uep_tot_pkts[j][i],
                                            bs.tot_sent)

    plt.figure()
    plt.gca().set_yscale('log')

    for (j, per) in enumerate(udp_err_rates):
        s = sorted(zip(overheads[j],
                       uep_err_rates[j],
                       uep_err_counts[j],
                       uep_tot_pkts[j]),
                   key=operator.itemgetter(0))
        s = np.array(s)
        ohs = s[:,0]
        mib_per = np.array(list(s[:,1]))[:,0]
        lib_per = np.array(list(s[:,1]))[:,1]
        mib_errc = np.array(list(s[:,2]))[:,0]
        lib_errc = np.array(list(s[:,2]))[:,1]
        mib_npkts = np.array(list(s[:,3]))[:,0]
        lib_npkts = np.array(list(s[:,3]))[:,1]

        mib_per_ci = []
        for z,n in zip(mib_errc, mib_npkts):
            mib_per_ci.append(success_err(z, n))
        mib_per_ci = np.array(mib_per_ci)

        lib_per_ci = []
        for z,n in zip(lib_errc, lib_npkts):
            lib_per_ci.append(success_err(z, n))
        lib_per_ci = np.array(lib_per_ci)

        mibline = plt.plot(ohs,
                           mib_per,
                           marker='.',
                           linewidth=0.5,
                           linestyle='-',
                           label="MIB e = {:.0e}".format(per))[0]
        plt.plot(ohs,
                 lib_per,
                 color=mibline.get_color(),
                 marker='.',
                 linewidth=0.5,
                 linestyle='--',
                 label="LIB e = {:.0e}".format(per))

    plt.ylim(1e-3, 1)
    plt.xlim(0, 0.8 * (1.01))
    plt.xlabel('Overhead')
    plt.ylabel('PER')
    plt.legend()
    plt.grid()

    plt.savefig("plot_ber_iid.pdf", format="pdf")
    plt.show()
