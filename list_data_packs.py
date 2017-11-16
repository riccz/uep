from uep import *
from utils.aws import *

def load_data_prefix_2(prefix, filter_func=None):
    s3 = boto3.client('s3')
    resp = s3.list_objects_v2(Bucket='uep.zanol.eu',
                              Prefix=prefix)
    items = list()
    print("Found {:d} packs".format(len(resp['Contents'])))
    for obj in resp['Contents']:
        if filter_func is None or filter_func(obj):
            items.append((load_data(obj['Key']), obj['Key']))
    return items


items = load_data_prefix_2('uep_iid')

for i, k in items:
    legend_str = ("Ks={!s},"
                  "RFs={!s},"
                  "EF={:d},"
                  "c={:.2f},"
                  "delta={:.2f},"
                  "e={:.0e}").format(i.get('Ks', 'none'),
                                     i.get('RFs', 'none'),
                                     i.get('EF', -1),
                                     i.get('c', -1),
                                     i.get('delta', -1),
                                     i.get('iid_per', -1))
    print("Key = " + k)
    print("  timestamp = {!s}".format(datetime.datetime.fromtimestamp(i.get('timestamp', 0))))
    print("  " + legend_str)
    print()
