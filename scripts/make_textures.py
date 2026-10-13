#!/usr/bin/env python3
"""make_textures.py — procedurally generate seamless albedo + normal maps for
the Gazebo Sim (Fortress, ogre2 PBR) field-hospital world.

Fully offline and reproducible (fixed RNG seed), so the photorealistic world
needs no Fuel downloads. Writes 512x512 PNGs into media/materials/textures/:

    concrete_albedo/_normal   clean corridor + decon floor
    dirt_albedo/_normal       open ground / mud
    canvas_albedo/_normal     tent fabric
    metal_albedo/_normal      perimeter walls
    fabric_albedo             cot/bed linens

Normal maps are derived from each albedo's luminance as a height field, so the
PBR shading picks up surface relief under the directional light + shadows.
"""

import os

import numpy as np
from PIL import Image

SIZE = 512
RNG = np.random.default_rng(7)


def _fbm(shape, octaves=5, persistence=0.55):
    """Tiling fractal value noise in [0,1]."""
    h, w = shape
    out = np.zeros(shape, dtype=np.float64)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        freq = 2 ** o
        base = RNG.random((freq, freq))
        # bilinear upsample with wrap so the tile is seamless
        ys = (np.arange(h) / h * freq) % freq
        xs = (np.arange(w) / w * freq) % freq
        y0 = np.floor(ys).astype(int) % freq; y1 = (y0 + 1) % freq
        x0 = np.floor(xs).astype(int) % freq; x1 = (x0 + 1) % freq
        fy = (ys - np.floor(ys))[:, None]; fx = (xs - np.floor(xs))[None, :]
        top = base[np.ix_(y0, x0)] * (1 - fx) + base[np.ix_(y0, x1)] * fx
        bot = base[np.ix_(y1, x0)] * (1 - fx) + base[np.ix_(y1, x1)] * fx
        out += amp * (top * (1 - fy) + bot * fy)
        norm += amp; amp *= persistence
    out /= norm
    return out


def _normal_from_height(height, strength=2.0):
    gy, gx = np.gradient(height.astype(np.float64))
    nz = np.ones_like(height)
    nx, ny = -gx * strength, -gy * strength
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return ((n * 0.5 + 0.5) * 255).astype(np.uint8)


def _save(name, rgb, out_dir):
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(out_dir, name + '.png'))


def _tint(noise, base, contrast):
    g = (0.5 + (noise - 0.5) * contrast)[..., None]
    return np.clip(np.array(base) * g, 0, 255)


def generate(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --- concrete: fine speckle, light grey ---
    n = _fbm((SIZE, SIZE), octaves=6, persistence=0.5)
    speck = (RNG.random((SIZE, SIZE)) > 0.995) * 0.25
    h = np.clip(n - speck, 0, 1)
    _save('concrete_albedo', _tint(h, (185, 185, 180), 0.35), out_dir)
    _save('concrete_normal', _normal_from_height(h, 1.2), out_dir)

    # --- dirt: warmer, rougher ---
    n = _fbm((SIZE, SIZE), octaves=6, persistence=0.62)
    _save('dirt_albedo', _tint(n, (120, 90, 60), 0.7), out_dir)
    _save('dirt_normal', _normal_from_height(n, 2.5), out_dir)

    # --- canvas: woven directional weave ---
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    weave = (0.5 + 0.5 * np.sin(xx / 3.0)) * (0.5 + 0.5 * np.sin(yy / 3.0))
    n = 0.7 * weave + 0.3 * _fbm((SIZE, SIZE), octaves=4)
    _save('canvas_albedo', _tint(n, (225, 222, 210), 0.18), out_dir)
    _save('canvas_normal', _normal_from_height(n, 1.0), out_dir)

    # --- corrugated metal: vertical ribs + grime ---
    ribs = 0.5 + 0.5 * np.sin(xx / 8.0)
    grime = _fbm((SIZE, SIZE), octaves=5)
    h = np.clip(0.75 * ribs + 0.25 * grime, 0, 1)
    _save('metal_albedo', _tint(h, (150, 155, 160), 0.45), out_dir)
    _save('metal_normal', _normal_from_height(h, 2.0), out_dir)

    # --- fabric: soft cloth for cots ---
    n = _fbm((SIZE, SIZE), octaves=5, persistence=0.5)
    _save('fabric_albedo', _tint(n, (235, 238, 242), 0.12), out_dir)

    # --- grass: green patchy ground, used tiled on the terrain ---
    blades = _fbm((SIZE, SIZE), octaves=7, persistence=0.6)
    patch = _fbm((SIZE, SIZE), octaves=2, persistence=0.7)   # large dry patches
    g = blades
    rgb = np.zeros((SIZE, SIZE, 3))
    # blend lush green -> dry yellow-green by the large-scale patch field
    lush = np.array([70, 120, 55]); dry = np.array([120, 130, 70])
    base = lush[None, None] * (1 - patch[..., None]) + dry[None, None] * patch[..., None]
    rgb = np.clip(base * (0.6 + 0.5 * g[..., None]), 0, 255)
    _save('grass_albedo', rgb, out_dir)
    _save('grass_normal', _normal_from_height(g, 1.6), out_dir)

    print(f'Wrote textures -> {out_dir}')


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    generate(os.path.join(os.path.dirname(here), 'media', 'materials', 'textures'))
