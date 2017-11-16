import concurrent.futures as cf
import datetime
import lzma
import math
import multiprocessing
import os
import os.path
import pathlib
import pickle
import random
import re
import sys

import boto3
import botocore

from dateutil.tz import tzutc

def getUrls(*arg):
    s3 = boto3.client('s3')
    fromDate = datetime.datetime(2017, 11, 5, 0, 0, tzinfo=tzutc())
    if (len(arg)>1):
        fromDate = arg[1]
    req = s3.list_objects_v2(
        Bucket='uep.zanol.eu',
        Prefix=arg[0],
    )
    urls = [req['Contents'][i]['Key'] for i in range(0,req['KeyCount'])
        if req['Contents'][i]['LastModified']>fromDate]
    print('Imported '+str(len(urls))+' datasets since ',fromDate, ' with prefix '+arg[0])
    return urls

def save_plot(fname, key):
    s3 = boto3.client('s3')
    s3.upload_file(fname, 'uep.zanol.eu', key,
                   ExtraArgs={'ACL': 'public-read'})
    url = ("http://uep.zanol.eu."
           "s3.amazonaws.com/{!s}".format(key))
    return url

def save_data(key, **kwargs):
    s3 = boto3.client('s3')
    data = lzma.compress(pickle.dumps(kwargs))
    s3.put_object(Body=data,
                  Bucket='uep.zanol.eu',
                  Key=key,
                  ACL='public-read')
    url = ("http://uep.zanol.eu.s3"
           ".amazonaws.com/{!s}".format(key))
    return url


def load_data(key):
    path = pathlib.Path.home().joinpath(".cache", "uep_load_data", key)
    try:
        bindata = open(str(path), 'rb').read()
    except FileNotFoundError as e:
        config = botocore.client.Config(read_timeout=300)
        s3 = boto3.client('s3', config=config)
        obj = s3.get_object(Bucket='uep.zanol.eu',
                            Key=key)
        bindata = obj['Body'].read()
        path.parent.mkdir(parents=True, exist_ok=True)
        open(str(path), 'xb').write(bindata)

    return pickle.loads(lzma.decompress(bindata))

def load_data_prefix_2(prefix, filter_func=None):
    """Get all S3 objects matching `prefix`. They are returned as a list
    of (body, key) pairs. If filter_func is not None, only the objects
    for which filter_func(obj) is True are returned. The filter_func
    is called with objects of type boto3.S3.ObjectSummary.

    """

    bucket = boto3.resource('s3').Bucket('uep.zanol.eu')
    items = list()
    for obj in bucket.objects.filter(Prefix=prefix):
        if filter_func is None or filter_func(obj):
            items.append((load_data(obj.key), obj.key))
    return items

def load_data_prefix(prefix, filter_func=None):
    """Same as load_data_prefix_2, but return only a list of object
    bodies.

    """

    return [i[0] for i in load_data_prefix_2(prefix, filter_func)]

def load_data_local(filename):
    data = lzma.decompress(open(filename,'rb').read())
    return pickle.loads(data)

def send_notification(msg):
    sns = boto3.client('sns', region_name='us-east-1')

    sns.publish(TopicArn='arn:aws:sns:us-east-1:402432167722:NotifyMe',
                Message=msg)
