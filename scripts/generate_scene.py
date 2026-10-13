#!/usr/bin/env python3
"""Generate every field-hospital artifact (Gazebo Sim world, GT grids, risk PGMs,
scene metadata) from field_hospital_scene.build_scene().

Run it after editing the SCENE block to regenerate a consistent set:

    python3 scripts/generate_scene.py                 # writes into ../maps + ../worlds
    python3 scripts/generate_scene.py --out /tmp/x    # custom output dirs

Importable either as an installed package (field_hospital_simulation) or directly from the
source tree, so it works before the first colcon build.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)          # src/field_hospital_simulation
sys.path.insert(0, _PKG_ROOT)

from field_hospital_simulation import field_hospital_scene as fhs   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps-dir',  default=os.path.join(_PKG_ROOT, 'maps'))
    ap.add_argument('--world-dir', default=os.path.join(_PKG_ROOT, 'worlds'))
    ap.add_argument('--tent-model', default='military_tent',
                    help='model:// name (a dir under models/) used for ward '
                         'tents in the Gazebo Sim world. Default: military_tent. '
                         'Others: medical_tent, tent (Fuel), or fetch with '
                         'scripts/fetch_fuel_model.py.')
    ap.add_argument('--seed', type=int, default=fhs.SEED,
                    help='RNG seed for the randomized terrain-patch layout. '
                         'Same seed => identical terrain (so all fusion methods '
                         'compare on the same terrain); different seed => '
                         'different layout. Default: %(default)s.')
    ap.add_argument('--crowd-seed', type=int, default=None,
                    help='Separate RNG seed for the crowd random walks. Omit '
                         '(default) to seed from OS entropy => a fresh random '
                         'crowd every generation. Pass an int to make the crowd '
                         'reproducible (e.g. the Monte-Carlo harness pins it to '
                         'the run seed so the paired comparison sees the same '
                         'crowd per seed).')
    ap.add_argument('--risk-density', type=float, default=1.0,
                    help='Scale factor on the terrain- and slope-patch counts '
                         '(1.0 = baseline mix). Higher packs in more risk and '
                         'more terrain/elevation overlap (the fusion-sensitive '
                         'regime); ~2.5 is the practical ceiling before patches '
                         'no longer fit. Default: %(default)s.')
    args = ap.parse_args()

    os.makedirs(args.maps_dir, exist_ok=True)
    os.makedirs(args.world_dir, exist_ok=True)

    meta, world_gz = fhs.generate(args.maps_dir, args.tent_model,
                                  args.seed, args.crowd_seed,
                                  args.risk_density)
    world_gz_path = os.path.join(args.world_dir, 'field_hospital_gz.sdf')
    with open(world_gz_path, 'w') as f:
        f.write(world_gz)

    # Procedural PBR textures for the Gazebo Sim world (offline, reproducible).
    try:
        from make_textures import generate as gen_tex
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'make_textures', os.path.join(_HERE, 'make_textures.py'))
        mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)
        gen_tex = mt.generate
    gen_tex(os.path.join(_PKG_ROOT, 'media', 'materials', 'textures'))

    print(f'Grid: {meta["width"]}x{meta["height"]} @ {meta["resolution"]} m/cell, '
          f'origin ({meta["origin_x"]}, {meta["origin_y"]})')
    print(f'Wrote maps     -> {args.maps_dir}')
    print(f'Wrote world    -> {world_gz_path}  (Gazebo Sim / Fortress PBR)')
    print('Suggested goals:')
    for g in meta['goals']:
        print(f'  {g["name"]:>18}: ({g["x"]}, {g["y"]})  {g["note"]}')


if __name__ == '__main__':
    main()
