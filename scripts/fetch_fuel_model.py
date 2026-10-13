#!/usr/bin/env python3
"""fetch_fuel_model.py — download a Gazebo Fuel model into the package models/
dir so it can be referenced as model://<dir> in the generated world.

  # browse/search Fuel:  https://app.gazebosim.org/fuel/models
  python3 scripts/fetch_fuel_model.py OpenRobotics "Relief Tent"            # auto dir name
  python3 scripts/fetch_fuel_model.py OpenRobotics "Relief Tent" --as tent  # force dir name

After fetching, regenerate the world pointing tents at it:
  python3 scripts/generate_scene.py --tent-model <dir>
  colcon build --packages-select field_hospital_simulation

NOTE: model:// resolves by the model's *internal* name. This tool keeps the dir
name aligned with that name unless you pass --as; if you override --as, make
sure the model's own meshes use relative URIs (most do) or it won't find them.
"""

import argparse
import io
import os
import sys
import urllib.parse
import urllib.request
import zipfile

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(_PKG_ROOT, 'models')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('owner')
    ap.add_argument('name', help='Fuel model name (quote if it has spaces)')
    ap.add_argument('--as', dest='dirname', default=None,
                    help='directory name under models/ (default: model name)')
    ap.add_argument('--version', default='tip')
    args = ap.parse_args()

    enc = urllib.parse.quote(args.name)
    url = (f'https://fuel.gazebosim.org/1.0/{args.owner}/models/'
           f'{enc}/{args.version}/{enc}.zip')
    dirname = args.dirname or args.name.replace(' ', '_')
    dest = os.path.join(MODELS, dirname)

    print(f'Downloading {args.owner}/{args.name} ...\n  {url}')
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
    except Exception as e:
        sys.exit(f'Download failed: {e}')
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(dest)
    print(f'Extracted -> {dest}')
    # report the internal model name so the user knows the model:// to use
    cfg = os.path.join(dest, 'model.config')
    if os.path.isfile(cfg):
        import re
        m = re.search(r'<name>([^<]+)</name>', open(cfg).read())
        if m:
            print(f'Internal model name: "{m.group(1)}"  '
                  f'(use --as "{m.group(1)}" if model:// fails to resolve)')


if __name__ == '__main__':
    main()
