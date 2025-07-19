import distutils
import json
import sys

def parse_version(s):
    return tuple(map(int, s.lstrip('v').split('.')))

data = json.load(sys.stdin)
versions = (v['name'] for v in data.values() if v['is_released'])
try:
    latest_version = max(versions, key=lambda v: parse_version(v))
except ValueError as e:
    if e.args and e.args[0] == 'max() arg is an empty sequence':
        latest_version = ''
    else:
        raise
print(latest_version)
