import datetime
import decimal
import glob
import lzma
import math
import os
import os.path
import pickle
import random
import re
import subprocess
import sys
import time

import boto3
import botocore
import numpy as np

from boto3.dynamodb.conditions import Attr
from decimal import Decimal
from subprocess import check_call, check_output, Popen, call

from ber import *

PROGRESS = {"BUILD": "FAILED",
            "PREPARE_VIDEO": "FAILED",
            "RUN": "FAILED",
            "PROCESS_OUTPUT": "FAILED",
            "UPLOADED_FILES": []}

def send_aws_mail(msg):
    global PROGRESS
    run_data = PROGRESS

    if run_data is not None:
        msg += "\n"
        msg += "BUILD = {!s}\n".format(run_data["BUILD"])
        msg += "PREPARE_VIDEO = {!s}\n".format(run_data["PREPARE_VIDEO"])
        msg += "RUN = {!s}\n".format(run_data["RUN"])
        msg += "PROCESS_OUTPUT = {!s}\n".format(run_data["PROCESS_OUTPUT"])

        # msg += "--- UPLOADED_FILES ---\n"
        # for f in run_data["UPLOADED_FILES"]:
        #     msg += "{!s}: {!s}\n".format(f[0], f[1])

    client = boto3.client('sns', region_name='us-east-1')
    r = client.publish(TopicArn='arn:aws:sns:us-east-1:402432167722:NotifyMe',
                       Message=msg)
    if not r['MessageId']: raise RuntimeError("SNS publish failed")

def prepare_video():
    global PROGRESS

    if os.path.exists('uep/dataset/stefan_cif_long.264'):
        print("Dataset already in place")
        PROGRESS['PREPARE_VIDEO'] = "CACHED"
        return

    config = botocore.client.Config(read_timeout=300)
    s3 = boto3.resource('s3', region_name='us-east-1', config=config)
    bucket = s3.Bucket('uep.zanol.eu')

    # Download if exists
    try:
        bucket.download_file('dataset_4000.tar.xz', 'uep/dataset.tar.xz')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            exists = False
        else:
            raise
    else:
        exists = True

    if not exists: # Generate the dataset
        check_call('''
        set -e
        cd uep/dataset
        wget 'http://trace.eas.asu.edu/yuv/stefan/stefan_cif.7z'
        7zr x stefan_cif.7z
        cd ../h264_scripts
        ./encode.sh
        cd ..
        tar -cJf dataset.tar.xz dataset/
        ''', shell=True)
        bucket.upload_file('uep/dataset.tar.xz', 'dataset.tar.xz')
        print("Uploaded dataset")
        PROGRESS['PREPARE_VIDEO'] = "OK"
    else:
        check_call('''
        set -e
        cd uep
        tar -xJf dataset.tar.xz
        ''', shell=True)
        print("Extracted downloaded dataset")
        PROGRESS['PREPARE_VIDEO'] = "CACHED"
    os.remove('uep/dataset.tar.xz')

def run():
    global PROGRESS

    os.chdir("uep/build/bin")
    git_sha1 = subprocess.check_output(["git", "log",
                                        "-1",
                                        "--format=%H"])
    git_sha1 = git_sha1.strip().decode()

    s3 = boto3.resource('s3', region_name='us-east-1')
    dynamo = boto3.resource('dynamodb', region_name='us-east-1')
    sim_tab = dynamo.Table('uep_sim_results')

    k0 = 100
    k1 = 1900
    rf0 = 5
    rf1 = 1
    ef = 2
    pktsize = 388
    send_rate = 1000000
    stream_name = 'stefan_cif_long'

    iid_err_ps = [0.01, 0.1, 0.3]
    overheads = np.linspace(0, 0.8, 8).tolist()
    print("Run with overheads =\n {!s}\n"
          "and iid_err_ps =\n {!s}".format(overheads, iid_err_ps))
    for (j, p) in enumerate(iid_err_ps):
        for (i, oh) in enumerate(overheads):
            n = math.ceil((k0+k1) * (1 + oh))
            srv_clog = open("server_console.log", "wt")
            srv_tcp_port = 12312 + i
            srv_proc = Popen(["./server",
                  "-p", str(srv_tcp_port),
                  "-K", "[{:d}, {:d}]".format(k0, k1),
                  "-R", "[{:d}, {:d}]".format(rf0, rf1),
                  "-E", str(ef),
                  "-L", str(pktsize),
                  "-n", str(n),
                  "-r", str(send_rate)],
                 stdout=srv_clog,
                 stderr=srv_clog)

            time.sleep(10)

            clt_clog = open("client_console.log", "wt")
            clt_udp_port = 12345 + i
            clt_proc = Popen(["./client",
                  "-l", str(clt_udp_port),
                  "-r", str(srv_tcp_port),
                  "-n", stream_name,
                  "-p", str(p),
                  "-t", "10"],
                 stdout=clt_clog,
                 stderr=clt_clog)

            if not (clt_proc.wait() == 0):
                raise subprocess.CalledProcessError(clt_proc.returncode,
                                                    clt_proc.args)
            if not (srv_proc.wait() == 0):
                raise subprocess.CalledProcessError(srv_proc.returncode,
                                                    srv_proc.args)
            print("Run ({:d}/{:d})"
                  " with overhead={:f},"
                  " iid err.p={:e},"
                  " (n={:d}) OK".format(j*len(overheads) + i + 1,
                                        len(overheads)*len(iid_err_ps),
                                        oh,
                                        p,
                                        n))

            bs = ber_scanner("server.log", "client.log")
            bs.scan()

            for logfile in ["server.log", "client.log"]:
                check_call(["tar", "-cJf", logfile + ".tar.xz", logfile])

            newid = random.getrandbits(64)
            ts = datetime.datetime.now().timestamp()
            bs_dump = pickle.dumps(bs)
            lzout = lzma.compress(bs_dump)

            # s3.meta.client.upload_file('server.log.tar.xz',
            #                            'uep.zanol.eu',
            #                            "sim_logs/{:d}_server.log.tar.xz".format(newid))
            # s3.meta.client.upload_file('client.log.tar.xz',
            #                            'uep.zanol.eu',
            #                            "sim_logs/{:d}_client.log.tar.xz".format(newid
            # ))
            s3.meta.client.put_object(Body=lzout,
                                      Bucket='uep.zanol.eu',
                                      Key="sim_scans/{:d}_ber_scanner.pickle.xz".format(newid))

            sim_tab.put_item(Item={'result_id': newid,
                                   'timestamp': Decimal(ts).quantize(Decimal('1e-3')),
                                   'git_commit': git_sha1,
                                   's3_server_log': "sim_logs/{:d}_server.log.tar.xz".format(newid),
                                   's3_client_log': "sim_logs/{:d}_client.log.tar.xz".format(newid),
                                   's3_ber_scanner': "sim_scans/{:d}_ber_scanner.pickle.xz".format(newid),
                                   'iid_per': Decimal(p).quantize(Decimal('1e-8')),
                                   'overhead': Decimal(oh).quantize(Decimal('1e-4')),
                                   'k0': k0,
                                   'k1': k1,
                                   'rf0': rf0,
                                   'rf1': rf1,
                                   'ef': ef,
                                   'pktsize': pktsize,
                                   'n': n,
                                   'send_rate': Decimal(send_rate).quantize(Decimal('1e2')),
                                   'stream_name': stream_name},
                             ConditionExpression=Attr('result_id').not_exists())

            for logfile in ["server.log", "client.log"]:
               os.remove(logfile + ".tar.xz")

    os.chdir("../..")
    PROGRESS['RUN'] = "OK"

def process_output():
    global PROGRESS

    check_call('''
    set -e
    cd uep/h264_scripts
    ./decode.sh
    ''', shell=True)

    # Upload videos
    check_call('''
    set -e
    cd uep/dataset_client
    aws s3 cp "stefan_cif_long.264" s3://uep.zanol.eu/simulation_videos/"$subdir_name"/"stefan_cif_long.264"
    ''', shell=True)

    PROGRESS['PROCESS_OUTPUT'] = "OK"

if __name__ == "__main__":
    try:
        prepare_video()
        run()
        # process_output()
        send_aws_mail("run_ber completed")

        # Shutdown
        check_call(["sudo", "poweroff"])

    except Exception as e:
        send_aws_mail("run_ber failed")
