#!/usr/bin/env python3
"""field_hospital_scene.py — single source of truth for the outbreak
field-hospital simulation.

One scene definition is rasterised into every artifact the sim needs, so the
Gazebo visuals, the ground-truth obstacle grid, the ground-truth terrain-cost
grid and the static terrain risk map are all spatially consistent by
construction:

    field_hospital_gz.sdf  Gazebo Sim (Fortress) world: physics + visuals + GT pose
    obstacle_gt.npy        what sim_obstacle_node reveals -> /octomap_grid
    terrain_gt.npy         what sim_terrain_node reveals  -> /risk/terrain
    terrain_slope.pgm      static /risk/elevation
    scene_meta.json        origin/res/goals shared by the nodes & launch

Coordinate convention (map frame):
    The robot starts at (0, 0) facing +x and drives into the compound (+x).
    Grid row 0 == minimum y (south); col 0 == minimum x (west). This matches
    the south-to-north PGM layout that elevation_map_node expects, and the
    row-major nav_msgs/OccupancyGrid layout, so no axis flips are ever needed.

Everything below the SCENE block is generic rasterisation/serialisation; edit
the SCENE block to reshape the hospital for your paper.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Grid framing
# ---------------------------------------------------------------------------
RES = 0.10                      # metres / cell for the GT grids and PGMs
X0, X1 = -5.0, 28.5             # world x extent (metres)
Y0, Y1 = -15.0, 15.0            # world y extent (metres) — wide open grass on
                                # the N/S sides of the tents (tents span to ±9.8,
                                # so this leaves ~15 m of drivable grass margin
                                # on each side)
WIDTH  = int(round((X1 - X0) / RES))   # columns  (x)
HEIGHT = int(round((Y1 - Y0) / RES))   # rows     (y)

# Ward-tent layout. Each tent is TENT_W (x) by TENT_L (y); the two columns sit
# at TENT_COLS x-centres with rows at y = ±TENT_Y. Widen the gap between the two
# TENT_COLS values to space the columns further apart (keep the east column's
# right edge < X1 so it stays on the map).
TENT_W, TENT_L, TENT_Y = 11.0, 6.6, 6.5
TENT_COLS = (5.5, 21.0)

# Terrain traversability costs (0..100), aligned with the class costs in
# terrain_map_node.TRAVERSABILITY_COST so a switch to real SegFormer matches.
T_FLOOR  = 5      # clean tent / corridor flooring
T_PATH   = 10     # packed walkway / aisle
T_GRAVEL = 40     # loose decon gravel
T_GRASS  = 5      # open ground inside the compound
T_MUD    = 50     # churned mud
T_BUSH   = 70     # low vegetation / scrub — snags wheels, hard to push through
T_ROCK   = 90     # rubble / rocky ground — very high traversal risk (still soft,
                  # not a hard wall, so the planner may cross it under study)


@dataclass
class Box:
    """An axis-aligned obstacle. `kind` drives both colour and how
    sim_obstacle_node reveals it: 'structural' is part of the known floor
    plan (revealed everywhere), 'movable' (cots, carts, people) is only
    revealed within sensor range, forcing reactive local avoidance."""
    cx: float
    cy: float
    sx: float
    sy: float
    sz: float
    name: str
    kind: str = 'structural'     # 'structural' | 'movable'
    shape: str = 'box'           # 'box' | 'cylinder'
    rgba: Tuple[float, float, float, float] = (0.55, 0.55, 0.55, 1.0)
    # If False the box is visual-only: NOT written into the obstacle/terrain GT
    # so the robot can drive through its footprint (e.g. enterable tents — the
    # canvas is just dressing; only the beds inside are real obstacles).
    blocks: bool = True


@dataclass
class Zone:
    """A painted rectangle. Later zones override earlier ones where they
    overlap, so paint broad bands first and details last."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    terrain: int                  # terrain cost 0..100 for /risk/terrain
    rgba: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # floor tint
    label: str = ''


@dataclass
class Scene:
    boxes: List[Box] = field(default_factory=list)
    zones: List[Zone] = field(default_factory=list)
    goals: List[dict] = field(default_factory=list)
    # Initial poses of the walking crowd. These get person models in the world
    # (driven at runtime by sim_crowd_node) and are written into scene_meta so
    # the node knows the names + start positions. NOT baked into the GT grids —
    # the crowd is dynamic and injected live, see sim_crowd_node.
    crowd: List[dict] = field(default_factory=list)


# Reproducible randomness for the scattered terrain patches and crowd spawn.
SEED = 7
CROWD_SIZE = 4
# Number of random slope (elevation-risk) patches scattered into the static
# elevation field per seed (see _terrain_slope_field).
N_SLOPE_PATCHES = 3
# Walking speed (m/s) baked into the crowd waypoint times. The Gazebo actor
# animation and sim_crowd_node's obstacle-grid replay both read those times, so
# this single value sets how fast the people move in sim. Lower = slower.
CROWD_SPEED = 0.4


# ===========================================================================
# SCENE DEFINITION — edit here to reshape the field hospital
# ===========================================================================
def tent_door_sign(row: str, col: int) -> int:
    """Which SHORT end each tent's door (and the mesh's open gable) is on:
        +1 = the +x short end,   -1 = the -x short end.

    We align this explicitly with your simulator's coordinate orientation.
    All ward tents open on the -x short end so every door faces the entrance
    corridor / robot start.
    """
    return -1


def _tent_keepouts(margin: float = 0.5):
    """Bounding rects of the four ward tents (+margin) — used to keep scattered
    patches and crowd spawns out of the structures."""
    tent_w, tent_l, tent_y = TENT_W, TENT_L, TENT_Y
    outs = []
    for xc in TENT_COLS:
        for sgn in (1, -1):
            cy = sgn * tent_y
            outs.append((xc - tent_w / 2 - margin, xc + tent_w / 2 + margin,
                         cy - tent_l / 2 - margin, cy + tent_l / 2 + margin))
    return outs


def _box_keepouts(scene: Scene, margin: float = 1.0):
    """Keepout rects for every blocking box, expanded by `margin` m on each side."""
    outs = []
    for b in scene.boxes:
        if not b.blocks:
            continue
        outs.append((b.cx - b.sx / 2 - margin, b.cx + b.sx / 2 + margin,
                     b.cy - b.sy / 2 - margin, b.cy + b.sy / 2 + margin))
    return outs


def _rect_overlaps(a, b) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


def _scatter_terrain_patches(s: Scene, rng, density: float = 1.0) -> None:
    """Scatter random terrain patches (mud, gravel, rock, bush) over the open
    grass. Each overrides the grass terrain cost where it lands (so the terrain
    risk map and the Gazebo floor tint both reflect it) but the tent interiors
    and the robot start are kept clear. The layout is a deterministic function
    of the RNG seed, so it is reproducible — identical for every method run at
    the same seed, and different across seeds.

    `density` scales the per-type patch counts (1.0 = baseline mix); raising it
    packs more risk in and makes terrain∩elevation overlaps more likely (the
    regime where the fusion rule matters). Each patch is a fixed 1x1 m
    square; only the count scales with density."""
    # Keep patches outside tent footprints and off the robot start.
    # _tent_keepouts gives a clean rect per tent canvas — using _box_keepouts
    # (walls + beds) pushed bed keepouts 1m outside the south tents' north face,
    # creating a dead zone in the middle corridor that made patches never appear
    # near the south tents.
    keepouts = _tent_keepouts(margin=0.5) + [
        (-2.0, 2.0, -2.0, 2.0),     # robot start
    ]
    # (label, terrain cost, floor tint rgb, count) — increasing traversal risk
    # from gravel -> mud -> bush -> rock.
    specs = [
        ('gravel', T_GRAVEL, (0.55, 0.55, 0.52), 3),
        ('mud',    T_MUD,    (0.36, 0.26, 0.16), 3),
        ('bush',   T_BUSH,   (0.20, 0.45, 0.20), 3),
        ('rock',   T_ROCK,   (0.42, 0.40, 0.40), 3),
    ]
    placed = []
    for label, cost, rgb, count in specs:
        count = max(0, int(round(count * density)))
        n = tries = 0
        while n < count and tries < max(800, 200 * count):
            tries += 1
            w = h = 1.0                       # fixed 1x1 m square
            cx = rng.uniform(X0 + 0.5 + w / 2, X1 - 0.5 - w / 2)
            cy = rng.uniform(Y0 + 0.5 + h / 2, Y1 - 0.5 - h / 2)
            rect = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
            if any(_rect_overlaps(rect, k) for k in keepouts + placed):
                continue
            s.zones.append(Zone(rect[0], rect[1], rect[2], rect[3],
                                cost, (rgb[0], rgb[1], rgb[2], 0.9),
                                f'{label}_{n}'))
            placed.append(rect)
            n += 1


def _walkable(x: float, y: float, keepouts) -> bool:
    """A point is walkable if it's inside the compound and off the tents."""
    if not (X0 + 0.8 <= x <= X1 - 0.8 and Y0 + 0.8 <= y <= Y1 - 0.8):
        return False
    if math.hypot(x, y) < 2.0:                 # keep clear of the robot start
        return False
    return not any(k[0] <= x <= k[1] and k[2] <= y <= k[3] for k in keepouts)


def _segment_clear(x0, y0, x1, y1, keepouts, step: float = 0.5) -> bool:
    """True if the straight leg (x0,y0)->(x1,y1) stays walkable the whole way
    (sampled every `step` m). Stops a person from walking through a tent wall or
    across a door corridor when wandering between two far-apart points."""
    d = math.hypot(x1 - x0, y1 - y0)
    nsteps = max(1, int(d / step))
    for k in range(nsteps + 1):
        f = k / nsteps
        if not _walkable(x0 + f * (x1 - x0), y0 + f * (y1 - y0), keepouts):
            return False
    return True


def _ang_diff(a: float, b: float) -> float:
    """Signed difference a-b wrapped to (-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _random_walk_waypoints(rng, keepouts, n_legs: int = 30,
                           speed: float = CROWD_SPEED, turn_time: float = 0.3):
    """A long, non-repeating random walk over the WHOLE walkable compound. Each
    leg heads to a uniformly random walkable point anywhere in the field, with a
    clear straight line to it (so the person stays in the open corridors, out of
    the tents and off the door approaches). Unlike the old fixed home-loop, the
    destinations span the entire field — people roam everywhere instead of
    circling one spot — and the trajectory is long enough (n_legs) that its
    period far exceeds a run, so it never visibly repeats within a run.

    Orientation is held CONSTANT for the duration of each straight leg and only
    turns — a brief `turn_time`-second in-place pivot — at the vertex between
    two legs. Gazebo's actor trajectory interpolates both position and
    orientation between consecutive waypoints; if each waypoint's yaw were
    instead set to its UPCOMING leg's heading (one entry per vertex, the
    simpler scheme), the body keeps rotating toward the NEXT turn for the
    whole CURRENT leg while still translating in the OLD direction — looking
    like it's walking sideways/backwards whenever the turn is sharp, which
    with fully random destinations is most of the time."""
    def _rand_pt():
        for _ in range(800):
            x = rng.uniform(X0 + 1.0, X1 - 1.0)
            y = rng.uniform(Y0 + 1.0, Y1 - 1.0)
            if _walkable(x, y, keepouts):
                return x, y
        return 0.0, 0.0                          # fallback (open field => rare)

    pts = [_rand_pt()]
    for _ in range(n_legs):
        cx, cy = pts[-1]
        for _try in range(500):
            x = rng.uniform(X0 + 1.0, X1 - 1.0)
            y = rng.uniform(Y0 + 1.0, Y1 - 1.0)
            if _walkable(x, y, keepouts) and _segment_clear(cx, cy, x, y, keepouts):
                pts.append((x, y))
                break
        else:
            pts.append((cx, cy))                 # no clear leg found: pause here
    pts.append(pts[0])                           # seamless loop close (far future)

    headings = [math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
               for i in range(len(pts) - 1)]

    wps, t, prev_yaw = [], 0.0, headings[0]
    for i in range(len(pts) - 1):
        x, y = pts[i]
        yaw = headings[i]
        if i > 0 and abs(_ang_diff(yaw, prev_yaw)) > 1e-3:
            # Stationary pivot: same position as the upcoming leg's start,
            # orientation still facing the leg just finished.
            wps.append([round(t, 2), round(x, 2), round(y, 2), round(prev_yaw, 3)])
            t += turn_time
        wps.append([round(t, 2), round(x, 2), round(y, 2), round(yaw, 3)])
        nx, ny = pts[i + 1]
        t += max(0.6, math.hypot(nx - x, ny - y) / speed)
        prev_yaw = yaw

    # Close the loop: arrive back at the start position (still facing the
    # closing leg's heading), then pivot in place to the first leg's heading
    # so the restart at wps[0] is a perfect pose match — no snap-rotation or
    # position jump at the seam.
    x, y = pts[-1]
    wps.append([round(t, 2), round(x, 2), round(y, 2), round(prev_yaw, 3)])
    if abs(_ang_diff(headings[0], prev_yaw)) > 1e-3:
        t += turn_time
        wps.append([round(t, 2), round(x, 2), round(y, 2), round(headings[0], 3)])
    return wps


def _door_keepouts(approach: float = 2.5, halfspan: float = 1.5):
    """Rects covering each tent's door opening + an approach corridor in front
    of it. Added to the CROWD keepouts (not the terrain ones) so wandering
    people never spawn in or path across a door — which is what left someone
    blocking the robot's entry/exit every run."""
    outs = []
    for col, xc in enumerate(TENT_COLS):
        for sgn in (1, -1):
            row = 'n' if sgn > 0 else 's'
            ii = tent_door_sign(row, col)             # -1: door on the -x end
            door_x = xc + ii * TENT_W / 2.0
            x_out  = door_x + ii * approach           # extends away from the tent
            cyc = sgn * TENT_Y
            outs.append((min(door_x, x_out), max(door_x, x_out),
                         cyc - halfspan, cyc + halfspan))
    return outs


def _scatter_crowd(rng, n: int = CROWD_SIZE):
    """Give each of n people a long random walk over the WHOLE walkable compound
    (see _random_walk_waypoints): uniformly random destinations, clear straight
    legs, no fixed home loop — so they roam everywhere and never circle one
    spot, and never sit in a tent door blocking the robot. Spawns are spread out
    (>= 1.5 m apart). With the default (entropy-seeded) crowd RNG the walks
    differ every generation, so each run gets a fresh random crowd."""
    keepouts = _tent_keepouts(margin=1.0) + _door_keepouts()
    crowd = []
    for i in range(n):
        wps = _random_walk_waypoints(rng, keepouts)
        # Nudge the spawn off any other person's spawn (regenerate the walk if
        # it lands too close); a few tries is plenty in the open field.
        for _try in range(20):
            hx, hy = wps[0][1], wps[0][2]
            if all(math.hypot(hx - c['x'], hy - c['y']) > 1.5 for c in crowd):
                break
            wps = _random_walk_waypoints(rng, keepouts)
        hx, hy = wps[0][1], wps[0][2]
        crowd.append({'name': f'crowd_{i}',
                      'x': round(hx, 2), 'y': round(hy, 2),
                      'yaw': wps[0][3], 'waypoints': wps})
    return crowd


def build_scene(seed: int = SEED, crowd_seed=None,
                risk_density: float = 1.0) -> Scene:
    s = Scene()

    TENT_H = 2.78    # matches the medical_tent mesh height

    # --- Terrain zones -----------------------------------------------------
    # Terrain is grass everywhere by default (rasterise() fills the grid with
    # T_GRASS), and the ground already carries a grass texture from
    # _ground_tiles(). Only the scattered mud patches (added below in
    # _scatter_terrain_patches) are painted as zones, so they stand out on the
    # grass instead of being washed out by a full-field green tint slab.

    # --- Ward tents (structural footprints) --------------------------------
    tent_rgba_green = (0.85, 0.85, 0.80, 1.0)
    tent_rgba_red   = (0.80, 0.55, 0.50, 1.0)
    wallc = (0.55, 0.55, 0.55, 1.0)
    
    tent_w, tent_l = TENT_W, TENT_L
    tent_y = TENT_Y
    # DOOR = width of the short-end opening (in y). 1.6 m (40 cm narrower than
    # the original 2.0 m) while still fitting the wheelchair (footprint
    # half-extent ~0.51 m) plus clearance.
    WALL_T, DOOR = 0.5, 1.6
    # Guy-line / stake apron grown OUTWARD from the three solid walls (the pegs
    # and tensioned lines that anchor the canvas to the ground stick out past
    # the wall on the outside). The inner face stays put; only the outside
    # footprint grows, so the obstacle map keeps the robot clear of the stakes.
    # The door posts are deliberately left thin so the entrance isn't narrowed.
    STAKE = 0.3
    tents = [(col, xc, sgn) for col, xc in enumerate(TENT_COLS) for sgn in (1, -1)]
    col_mid = sum(TENT_COLS) / 2.0

    for col, xc, sgn in tents:
        rgba = tent_rgba_red if xc > col_mid else tent_rgba_green
        row  = 'n' if sgn > 0 else 's'
        name = f"tent_{row}_{col}"
        s.boxes.append(Box(xc, sgn * tent_y, tent_w, tent_l, TENT_H, name,
                            rgba=rgba, blocks=False))
        cyc = sgn * tent_y

        # We enforce a constant door sign direction vector relative to the map
        ii  = tent_door_sign(row, col)
        door_x  = xc + ii * tent_w / 2.0      # short end with the door
        outer_x = xc - ii * tent_w / 2.0      # far short end (solid)
        post = (tent_l - DOOR) / 2.0          # wall post each side of the opening

        # (cx, cy, sx, sy, tag, out) — `out` is the outward unit normal (away
        # from the tent centre) along the wall's thin axis; solid walls grow by
        # STAKE in that direction. Door posts pass out=None (left thin).
        walls = [
            (xc, cyc + tent_l / 2.0, tent_w, WALL_T, 'yplus', ( 0.0,  1.0)),  # long wall
            (xc, cyc - tent_l / 2.0, tent_w, WALL_T, 'yneg',  ( 0.0, -1.0)),  # long wall
            (outer_x, cyc, WALL_T, tent_l, 'outer', (float(-ii), 0.0)),       # far short end
            (door_x, cyc - DOOR / 2.0 - post / 2.0, WALL_T, post, 'd1', None),  # door post
            (door_x, cyc + DOOR / 2.0 + post / 2.0, WALL_T, post, 'd2', None),  # door post
        ]
        for wx, wy, wsx, wsy, tag, out in walls:
            if out is not None and STAKE > 0.0:
                nx, ny = out
                wsx += STAKE * abs(nx)
                wsy += STAKE * abs(ny)
                wx  += nx * STAKE / 2.0   # keep the inner face fixed, grow outward
                wy  += ny * STAKE / 2.0
            s.boxes.append(Box(wx, wy, wsx, wsy, TENT_H,
                               f'tentwall_{name}_{tag}', rgba=wallc))

    # --- Beds + IV stands inside the tents (real obstacles) ----------------
    cot_rgba    = (0.90, 0.90, 0.95, 1.0)
    iv_rgba     = (0.80, 0.80, 0.85, 1.0)
    person_rgba = (0.20, 0.35, 0.75, 1.0)
    bi = vi = 0
    # Beds run LENGTHWISE (long axis along x, the tent's 11 m dimension) and
    # line ONLY the +y long wall (the -y row was removed to open drivable space
    # on the south side of each tent). bed_w/2 + bedrow must stay inside |y|
    # ~2.2 m, where the ridge-tent canvas is still > the bed height (it drops
    # below 0.85 m past |y|~2.5 m, which is what made the old across-width beds
    # poke through the roof).
    bed_l, bed_w = 2.0, 0.9
    bedrow = 1.6

    for col, xc, sgn in tents:
        cyc = sgn * tent_y
        row = 'n' if sgn > 0 else 's'
        ii  = tent_door_sign(row=row, col=col)
        door_x = xc + ii * tent_w / 2.0

        # bed centres march from the door toward the solid end
        bed_xs = [door_x - ii * off for off in (2.5, 5.0, 7.5)]
        for bx in bed_xs:
            # Only the +y row (s_row = 1); the -y row was removed.
            for s_row in (1,):
                s.boxes.append(Box(bx, cyc + s_row * bedrow, bed_l, bed_w, 0.7,
                                   f'cot_{bi}', rgba=cot_rgba))
                bi += 1
        # IV cart parked in the aisle at the solid end
        s.boxes.append(Box(door_x - ii * 9.5, cyc, 0.5, 0.5, 1.6,
                           f'cart_{vi}', shape='cylinder', rgba=iv_rgba))
        vi += 1

    # Scattered terrain patches (mud/gravel/rock/bush) over the open ground use
    # the reproducible `seed` RNG: the same `seed` reproduces an identical
    # terrain layout (so all three fusion methods compare on the same terrain).
    rng = np.random.default_rng(seed)
    _scatter_terrain_patches(s, rng, risk_density)
    # The crowd uses an INDEPENDENT RNG so its random walks can vary without
    # touching the terrain. `crowd_seed=None` (the default) seeds from OS entropy
    # -> a fresh random crowd every generation (per-run randomization). Pass an
    # int crowd_seed to make the crowd reproducible (the Monte-Carlo harness does
    # this so the paired comparison still sees an identical crowd per seed).
    crowd_rng = np.random.default_rng(crowd_seed)
    s.crowd = _scatter_crowd(crowd_rng)

    # --- Mission goals: intake / intra-facility / egress --------------------
    # Three scenario types × four tents = 4 + 12 + 4 = 20 goals total.
    #
    # spawn:  tent column centre (TENT_COLS[col], ±tent_y) — open aisle,
    #         clear of beds, facing -x toward the compound entrance.
    # goal:   0.6 m inside the -x door, 0.5 m toward the cleared -y row,
    #         keeping the chair clear of the remaining +y beds.
    IN_YAW = 0.0    # face into the tent (+x) on arrival

    def _pt(x, y, yaw):
        return {'x': x, 'y': y, 'yaw': yaw}

    OUTSIDE = (-2.0, 0.0, 0.0)    # entrance / external staging area

    # (label, col_idx, row_sign): NW=col0/north, NE=col1/north,
    #                              SE=col1/south, SW=col0/south
    TENT_DEFS = [('nw', 0, +1), ('ne', 1, +1), ('se', 1, -1), ('sw', 0, -1)]

    def tent_goal(col, sgn):
        """Aisle drop-off: 0.6 m inside door, 0.5 m toward cleared -y row."""
        return round(TENT_COLS[col] + 0.6, 2), round(sgn * tent_y - 0.5, 2)

    def tent_spawn(col, sgn):
        """Column-centre spawn inside a tent, facing the compound entrance."""
        return _pt(TENT_COLS[col], sgn * tent_y, math.pi)

    s.goals = []

    # Intake (4): entrance -> each tent
    for tname, col, sgn in TENT_DEFS:
        gx_t, gy_t = tent_goal(col, sgn)
        s.goals.append({
            'name': f'intake_to_{tname}', 'scenario': 'intake',
            'start': _pt(*OUTSIDE),
            'x': gx_t, 'y': gy_t, 'yaw': IN_YAW,
            'note': f'Intake: entrance -> {tname.upper()} ward tent.',
        })

    # Intra-facility (12): every ordered tent pair
    for src, sc, ss in TENT_DEFS:
        for dst, dc, ds in TENT_DEFS:
            if src == dst:
                continue
            gx_t, gy_t = tent_goal(dc, ds)
            s.goals.append({
                'name': f'intra_{src}_to_{dst}', 'scenario': 'intra',
                'start': tent_spawn(sc, ss),
                'x': gx_t, 'y': gy_t, 'yaw': IN_YAW,
                'note': f'Intra-facility: {src.upper()} -> {dst.upper()} ward tent.',
            })

    # Egress (4): each tent -> entrance
    for tname, col, sgn in TENT_DEFS:
        s.goals.append({
            'name': f'egress_from_{tname}', 'scenario': 'egress',
            'start': tent_spawn(col, sgn),
            'x': OUTSIDE[0], 'y': OUTSIDE[1], 'yaw': OUTSIDE[2],
            'note': f'Egress: {tname.upper()} ward tent -> entrance.',
        })

    return s


# ===========================================================================
# Rasterisation
# ===========================================================================
def _world_to_cell(x: float, y: float) -> Tuple[int, int]:
    c = int(math.floor((x - X0) / RES))
    r = int(math.floor((y - Y0) / RES))
    return r, c


def _rect_slice(xmin, xmax, ymin, ymax):
    r0, c0 = _world_to_cell(xmin, ymin)
    r1, c1 = _world_to_cell(xmax, ymax)
    r0 = max(0, min(HEIGHT, r0)); r1 = max(0, min(HEIGHT, r1))
    c0 = max(0, min(WIDTH,  c0)); c1 = max(0, min(WIDTH,  c1))
    return slice(r0, r1), slice(c0, c1)


def rasterise(scene: Scene):
    obstacle   = np.zeros((HEIGHT, WIDTH), dtype=np.int16)
    structural = np.zeros((HEIGHT, WIDTH), dtype=bool)
    terrain    = np.full((HEIGHT, WIDTH), T_GRASS, dtype=np.int16)

    for z in scene.zones:
        rs, cs = _rect_slice(z.xmin, z.xmax, z.ymin, z.ymax)
        terrain[rs, cs] = z.terrain

    for b in scene.boxes:
        if not b.blocks:
            continue
        rs, cs = _rect_slice(b.cx - b.sx / 2, b.cx + b.sx / 2,
                             b.cy - b.sy / 2, b.cy + b.sy / 2)
        obstacle[rs, cs] = 100
        if b.kind == 'structural':
            structural[rs, cs] = True

    return obstacle.astype(np.int8), terrain.astype(np.int8), structural


# ===========================================================================
# Serialisation helpers
# ===========================================================================
def _write_pgm(path: str, prob: np.ndarray) -> None:
    r = np.clip(prob, 0.0, 1.0)
    pix = np.round((1.0 - r) * 255.0).astype(np.uint8)
    h, w = pix.shape
    with open(path, 'wb') as f:
        f.write(b'P5\n')
        f.write(f'{w} {h}\n255\n'.encode())
        f.write(pix.tobytes())


def _write_map_yaml(path: str, pgm_name: str) -> None:
    lines = [
        f'image: {pgm_name}',
        f'resolution: {RES}',
        f'origin: [{X0}, {Y0}, 0.0]',
        'negate: 0',
        'occupied_thresh: 0.65',
        'free_thresh: 0.196',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _terrain_slope_field(scene: Scene, seed: int = SEED,
                         density: float = 1.0) -> np.ndarray:
    """Static elevation-risk field used as /risk/elevation in 'terrain' mode.

    Low baseline everywhere, one fixed moderate (0.45) band toward the hot end,
    plus N_SLOPE_PATCHES random elevated 'slope' patches scattered over the open
    ground. (The two old 0.85 bands spanning the whole compound were removed —
    they dominated the risk field and severed green<->red.)

    The patches use an INDEPENDENT RNG stream derived from `seed` (a spawned
    SeedSequence child, so it doesn't alias the terrain-patch positions), making
    them reproducible per seed — identical across the three fusion methods, like
    the terrain patches — while differing between seeds. Soft risk only: even
    the steepest patch is < mu_blocked, so the planner may cross it at cost."""
    risk = np.full((HEIGHT, WIDTH), 0.05, dtype=np.float32)
    # Fixed moderate elevated zone toward the hot (red) tent column.
    rs, cs = _rect_slice(18.0, 24.0, -9.0, 9.0)
    risk[rs, cs] = np.maximum(risk[rs, cs], 0.45)

    # Scatter random slope patches over the open ground (off the tents and the
    # robot start), magnitude 0.3..0.7, fixed 1x1 m. max-combine so they
    # stack onto the baseline/band rather than overwrite.
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    keepouts = _box_keepouts(scene, margin=1.0) + [(-2.0, 2.0, -2.0, 2.0)]
    placed = []
    n_patches = max(0, int(round(N_SLOPE_PATCHES * density)))
    n = tries = 0
    while n < n_patches and tries < max(800, 200 * n_patches):
        tries += 1
        w = h = 1.0                           # fixed 1x1 m square
        cx = rng.uniform(X0 + 0.5 + w / 2, X1 - 0.5 - w / 2)
        cy = rng.uniform(Y0 + 0.5 + h / 2, Y1 - 0.5 - h / 2)
        rect = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
        if any(_rect_overlaps(rect, k) for k in keepouts + placed):
            continue
        val = rng.uniform(0.3, 0.7)
        rrs, ccs = _rect_slice(rect[0], rect[1], rect[2], rect[3])
        risk[rrs, ccs] = np.maximum(risk[rrs, ccs], val)
        placed.append(rect)
        n += 1

    # Tents carry NO elevation: clamp every tent footprint back to the flat
    # baseline so neither the fixed band (which overlaps the red column) nor a
    # slope patch raises the clean, flat tent floor. Done last so it overrides.
    for (xmin, xmax, ymin, ymax) in _tent_keepouts(margin=0.0):
        rrs, ccs = _rect_slice(xmin, xmax, ymin, ymax)
        risk[rrs, ccs] = 0.05
    return risk


def generate(out_dir: str, tent_model: str = 'tent', seed: int = SEED,
             crowd_seed=None, risk_density: float = 1.0) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    scene = build_scene(seed, crowd_seed, risk_density)
    obstacle, terrain, structural = rasterise(scene)

    np.save(os.path.join(out_dir, 'obstacle_gt.npy'), obstacle)
    np.save(os.path.join(out_dir, 'obstacle_structural.npy'), structural)
    np.save(os.path.join(out_dir, 'terrain_gt.npy'), terrain)

    slope = _terrain_slope_field(scene, seed, risk_density)
    _write_pgm(os.path.join(out_dir, 'terrain_slope.pgm'), slope)
    _write_map_yaml(os.path.join(out_dir, 'terrain_slope.yaml'),
                    'terrain_slope.pgm')

    # Risk-layer density and the terrain∩elevation overlap — the multi-layer
    # conflict region where the fusion rule actually changes the fused value
    # (independent vs max vs logodds diverge only where >1 layer is elevated).
    # Logged into scene_meta so metrics_node records it per run.
    tmask = terrain != T_GRASS                  # terrain-patch cells
    smask = slope > 0.06                         # elevation band + slope patches
    overlap_cells = int(np.count_nonzero(tmask & smask))
    cell_area = RES * RES

    meta = {
        'resolution': RES,
        'origin_x': X0,
        'origin_y': Y0,
        'width': WIDTH,
        'height': HEIGHT,
        'robot_start': {'x': -2.0, 'y': 0.0, 'yaw': 0.0},
        'goals': scene.goals,
        'crowd': scene.crowd,
        'obstacle_gt': 'obstacle_gt.npy',
        'obstacle_structural': 'obstacle_structural.npy',
        'terrain_gt': 'terrain_gt.npy',
        'risk_density': float(risk_density),
        'terrain_patch_cells': int(np.count_nonzero(tmask)),
        'slope_elevated_cells': int(np.count_nonzero(smask)),
        'overlap_cells': overlap_cells,
        'overlap_m2': round(overlap_cells * cell_area, 3),
    }
    with open(os.path.join(out_dir, 'scene_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    from . import sdf_world_gz
    world_gz = sdf_world_gz.build_world_gz(scene, tent_model)
    return meta, world_gz