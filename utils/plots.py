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

def save_plot(my_plot, name, fmt):
    print_fig = my_plot
    print_fig.set_size_inches(10,8)
    print_fig.set_dpi(200)
    fullname = "average/{}.{}".format(name, fmt)
    os.makedirs(os.path.dirname(fullname), exist_ok=True)
    print_fig.savefig(fullname, format=fmt)

def save_plot_png(my_plot, name):
    save_plot(my_plot, name, 'png')

def save_plot_pdf(my_plot, name):
    save_plot(my_plot, name, 'pdf')
