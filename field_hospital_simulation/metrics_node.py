#!/usr/bin/env python3
"""metrics_node.py — per-run evaluation logger for the fusion Monte-Carlo study.

One run = one /goal_pose. The node arms on the goal, integrates the robot's
executed trajectory until it reaches the goal (success) or fails to get there
within stuck_timeout_s (failure, reason 'stuck'), then appends one
self-describing row to a CSV and dumps the
raw trajectory to an .npz for offline plots. Launch it alongside the stack with
the run's `method`/`seed`/`goal` parameters set, send a goal, repeat.

Why a fixed reference field
---------------------------
Every fusion method publishes a *different* /risk_fused, so integrating risk on
a method's own map would reward whichever rule underestimates risk most. All
spatial-risk metrics here are therefore evaluated on a single, method-
independent ground-truth field built from the static scene layers (terrain_gt,
the elevation/terrain-slope risk map, obstacle_gt) combined with the canonical
independent-failure rule. The method's own /risk_fused is sampled too, but only
as a *calibration* metric (own vs. GT), never as the safety score. The moving
crowd is dynamic, so it is scored separately (clearance + collisions from
/crowd_markers) rather than baked into the static field.

Subscriptions
    /odom           nav_msgs/Odometry          ground-truth pose + twist
    /cmd_vel        geometry_msgs/Twist         commanded velocity (jerk source)
    /goal_pose      geometry_msgs/PoseStamped   arms a run
    /crowd_markers  visualization_msgs/MarkerArray  live people positions
    /risk_fused     nav_msgs/OccupancyGrid      method's own fused risk (calib.)

Outputs (default ~/sar_metrics)
    runs.csv        one row per run (appended)
    <run_id>.npz    full trajectory time series for offline analysis
"""

import csv
import json
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray


# ---------------------------------------------------------------------------
# Static ground-truth risk field
# ---------------------------------------------------------------------------

def _read_pgm(path: str) -> np.ndarray:
    """Minimal P5/P2 PGM reader (skips comments). Returns uint8 HxW."""
    with open(path, 'rb') as f:
        data = f.read()
    # Tokenise the header, skipping '#' comment lines.
    tokens: List[bytes] = []
    i = 0
    n = len(data)
    magic = None
    while len(tokens) < 3 and i < n:
        while i < n and data[i:i + 1].isspace():
            i += 1
        if i < n and data[i:i + 1] == b'#':
            while i < n and data[i:i + 1] not in (b'\n', b'\r'):
                i += 1
            continue
        start = i
        while i < n and not data[i:i + 1].isspace():
            i += 1
        tok = data[start:i]
        if magic is None:
            magic = tok
        else:
            tokens.append(tok)
    width, height, maxval = (int(tokens[0]), int(tokens[1]), int(tokens[2]))
    i += 1  # single whitespace after maxval precedes the raster
    if magic == b'P5':
        raster = np.frombuffer(data[i:i + width * height], dtype=np.uint8)
        return raster.reshape(height, width)
    if magic == b'P2':
        vals = np.array(data[i:].split(), dtype=np.int32)[:width * height]
        return (vals.reshape(height, width) * 255 // maxval).astype(np.uint8)
    raise ValueError(f'unsupported PGM magic {magic!r} in {path}')


def _independent_combine(layers: List[np.ndarray]) -> np.ndarray:
    survival = np.ones_like(layers[0])
    for mu in layers:
        survival *= (1.0 - mu)
    return np.clip(1.0 - survival, 0.0, 1.0)


class GroundTruthField:
    """Static, method-independent risk field, indexed in map coordinates.

    Built once from disk: terrain_gt.npy (cost 0-100), the elevation/terrain-
    slope risk .pgm (risk 0-1, encoded pixel = (1-risk)*255), obstacle_gt.npy
    (0/100). All three share the scene grid (bottom-origin, row-major), so no
    flips are needed. Combined with the canonical independent-failure rule.
    """

    def __init__(self, maps_dir: str, meta: dict, elevation_pgm: str):
        self.res = float(meta['resolution'])
        self.ox = float(meta['origin_x'])
        self.oy = float(meta['origin_y'])
        self.h = int(meta['height'])
        self.w = int(meta['width'])

        terrain = np.load(os.path.join(maps_dir, meta['terrain_gt'])).astype(
            np.float32)
        self.terrain = np.clip(terrain, 0.0, 100.0) / 100.0
        obstacle = np.load(os.path.join(maps_dir, meta['obstacle_gt'])).astype(
            np.float32)
        self.obstacle = np.clip(obstacle, 0.0, 100.0) / 100.0  # 0/1

        if elevation_pgm and os.path.isfile(elevation_pgm):
            pix = _read_pgm(elevation_pgm).astype(np.float32)
            elev = 1.0 - pix / 255.0          # invert: dark = high risk
            elev = self._resample(elev, float(meta['resolution']))
        else:
            elev = np.zeros((self.h, self.w), dtype=np.float32)
        self.elevation = np.clip(elev, 0.0, 1.0)

        self.risk = _independent_combine(
            [self.terrain, self.elevation, self.obstacle]).astype(np.float32)

    def _resample(self, arr: np.ndarray, src_res: float) -> np.ndarray:
        if arr.shape == (self.h, self.w):
            return arr
        rows = np.clip(((np.arange(self.h) + 0.5) * self.res / src_res).astype(int),
                       0, arr.shape[0] - 1)
        cols = np.clip(((np.arange(self.w) + 0.5) * self.res / src_res).astype(int),
                       0, arr.shape[1] - 1)
        return arr[np.ix_(rows, cols)]

    def _rc(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        c = int((x - self.ox) / self.res)
        r = int((y - self.oy) / self.res)
        if 0 <= r < self.h and 0 <= c < self.w:
            return r, c
        return None

    def risk_at(self, x: float, y: float) -> float:
        rc = self._rc(x, y)
        return float(self.risk[rc]) if rc else 0.0

    def elevation_at(self, x: float, y: float) -> float:
        rc = self._rc(x, y)
        return float(self.elevation[rc]) if rc else 0.0

    def terrain_at(self, x: float, y: float) -> float:
        rc = self._rc(x, y)
        return float(self.terrain[rc]) if rc else 0.0

    def obstacle_at(self, x: float, y: float) -> float:
        rc = self._rc(x, y)
        return float(self.obstacle[rc]) if rc else 0.0

    def obstacle_contact(self, x: float, y: float, radius: float,
                         thr: float = 0.99) -> Tuple[bool, float]:
        """(in_contact, distance_to_nearest_obstacle_cell) within a search box."""
        rc = self._rc(x, y)
        if rc is None:
            return False, float('inf')
        rad = max(1, int(round(radius / self.res)))
        r, c = rc
        r0, r1 = max(0, r - rad - 2), min(self.h, r + rad + 3)
        c0, c1 = max(0, c - rad - 2), min(self.w, c + rad + 3)
        sub = self.obstacle[r0:r1, c0:c1] >= thr
        if not sub.any():
            return False, float('inf')
        rr, cc = np.nonzero(sub)
        dx = (cc + c0 - c) * self.res
        dy = (rr + r0 - r) * self.res
        d = float(np.sqrt(dx * dx + dy * dy).min())
        return d <= radius, d


# ---------------------------------------------------------------------------
# Per-run accumulator
# ---------------------------------------------------------------------------

class RunRecorder:
    """Holds the time series for one armed run and finalises the metrics."""

    def __init__(self, gt: GroundTruthField, goal: Tuple[float, float, float],
                 start: Tuple[float, float], robot_radius: float,
                 person_radius: float, high_risk_thr: float,
                 inflation_radius: float = 0.0):
        self.gt = gt
        self.goal = goal
        self.start = start
        self.robot_radius = robot_radius
        self.inflation_radius = inflation_radius
        self.contact_dist = robot_radius + person_radius
        self.high_risk_thr = high_risk_thr

        self.t: List[float] = []
        self.x: List[float] = []
        self.y: List[float] = []
        self.v: List[float] = []          # commanded/actual linear speed
        self.w: List[float] = []          # yaw rate
        self.risk: List[float] = []       # GT risk under the footprint
        self.elev: List[float] = []       # elevation/terrain-slope risk
        self.peak_layer: List[float] = [] # max across independent layers at each step
        self.own_risk: List[float] = []   # method's own /risk_fused at robot
        self.crowd_clear: List[float] = []  # min distance to any person
        self.obs_clear: List[float] = []    # distance to nearest GT obstacle

        self._crowd_in_contact = False
        self._obs_in_contact = False
        self.collisions_crowd = 0
        self.collisions_static = 0

    def add(self, t: float, x: float, y: float, v: float, w: float,
            own_risk: float, crowd_xy: List[Tuple[float, float]]) -> None:
        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.v.append(v)
        self.w.append(w)
        self.risk.append(self.gt.risk_at(x, y))
        elev = self.gt.elevation_at(x, y)
        self.elev.append(elev)
        self.peak_layer.append(max(
            self.gt.terrain_at(x, y), elev, self.gt.obstacle_at(x, y)))
        self.own_risk.append(own_risk)

        # Crowd clearance + rising-edge collision counting.
        if crowd_xy:
            dmin = min(math.hypot(x - px, y - py) for px, py in crowd_xy)
        else:
            dmin = float('inf')
        self.crowd_clear.append(dmin)
        contact = dmin <= self.contact_dist
        if contact and not self._crowd_in_contact:
            self.collisions_crowd += 1
        self._crowd_in_contact = contact

        # Static obstacle clearance + collisions.
        in_obs, dobs = self.gt.obstacle_contact(x, y, self.robot_radius)
        self.obs_clear.append(dobs)
        if in_obs and not self._obs_in_contact:
            self.collisions_static += 1
        self._obs_in_contact = in_obs

    # -- finalisation -------------------------------------------------------

    def _reached_goal(self, xy_tol: float) -> bool:
        if not self.x:
            return False
        return math.hypot(self.x[-1] - self.goal[0],
                          self.y[-1] - self.goal[1]) <= xy_tol

    def finalize(self, success: bool, reason: str, xy_tol: float) -> dict:
        t = np.asarray(self.t)
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        v = np.asarray(self.v)
        w = np.asarray(self.w)
        risk = np.asarray(self.risk)
        peak_layer = np.asarray(self.peak_layer)
        own = np.asarray(self.own_risk)

        if len(t) >= 2:
            dt = np.diff(t)
            dt = np.clip(dt, 1e-3, None)
            seg = np.hypot(np.diff(x), np.diff(y))
            path_len = float(seg.sum())
            duration = float(t[-1] - t[0])
            # Risk integrals on the GT field (time- and path-weighted).
            risk_mid = 0.5 * (risk[1:] + risk[:-1])
            cum_risk_time = float((risk_mid * dt).sum())
            cum_risk_path = float((risk_mid * seg).sum())
            own_mid = 0.5 * (own[1:] + own[:-1])
            cum_own_time = float((own_mid * dt).sum())
            # Signed bias: positive = method overestimates risk, negative = underestimates.
            risk_calibration_bias = float(np.nanmean(own - risk))
            # Smoothness: accel = dv/dt, jerk = d(accel)/dt.
            accel = np.diff(v) / dt
            jerk = np.diff(accel) / np.clip(dt[1:], 1e-3, None)
            ang_accel = np.diff(w) / dt
            control_effort_lin = float((np.abs(accel) * dt).sum())
            control_effort_ang = float((np.abs(ang_accel) * dt).sum())
            jerk_rms = float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else 0.0
            jerk_peak = float(np.max(np.abs(jerk))) if jerk.size else 0.0
            ang_jerk = (np.diff(ang_accel) / np.clip(dt[1:], 1e-3, None)
                        if ang_accel.size > 1 else np.array([]))
            ang_jerk_rms = (float(np.sqrt(np.mean(ang_jerk ** 2)))
                            if ang_jerk.size else 0.0)
            # Stops: rising edges of "near-zero speed".
            stopped = v < 0.02
            num_stops = int(np.sum(stopped[1:] & ~stopped[:-1]))
            high_dwell = float((dt[risk_mid > self.high_risk_thr]).sum())
            high_dwell_frac = high_dwell / duration if duration > 0 else 0.0
            # Per-layer peak > 0.3 dwell and worst value.
            peak_mid = 0.5 * (peak_layer[1:] + peak_layer[:-1])
            mask_03 = peak_mid > 0.3
            dwell_risk_03_s = float(dt[mask_03].sum())
            peak_risk_03 = float(peak_mid[mask_03].max()) if mask_03.any() else 0.0
        else:
            path_len = duration = cum_risk_time = cum_risk_path = 0.0
            cum_own_time = control_effort_lin = 0.0
            control_effort_ang = jerk_rms = jerk_peak = ang_jerk_rms = 0.0
            high_dwell_frac = dwell_risk_03_s = peak_risk_03 = 0.0
            risk_calibration_bias = float('nan')
            num_stops = 0

        straight = math.hypot(self.goal[0] - self.start[0],
                              self.goal[1] - self.start[1])
        spl = (straight / max(straight, path_len)) if (success and path_len > 1e-6) else 0.0

        crowd_clear = np.asarray(self.crowd_clear)
        obs_clear = np.asarray(self.obs_clear)
        # Surface clearances (body-to-body / robot-body-to-obstacle-surface).
        # Negative values mean overlap (collision registered at that step).
        crowd_surf_clear = crowd_clear - self.contact_dist
        obs_surf_clear = obs_clear - self.robot_radius + self.inflation_radius
        finite_surf_crowd = crowd_surf_clear[np.isfinite(crowd_surf_clear)]
        finite_surf_obs = obs_surf_clear[np.isfinite(obs_surf_clear)]
        # Hazard = nearest surface (person or obstacle), per-step minimum.
        hazard_surf_clear = np.minimum(crowd_surf_clear, obs_surf_clear)
        finite_haz = hazard_surf_clear[np.isfinite(hazard_surf_clear)]

        final_xy_err = (math.hypot(self.x[-1] - self.goal[0],
                                   self.y[-1] - self.goal[1])
                        if self.x else float('nan'))

        return {
            'success': int(bool(success)),
            'failure_reason': reason,
            'time_to_goal_s': round(duration, 3),
            'path_length_m': round(path_len, 3),
            'straight_line_m': round(straight, 3),
            'spl': round(spl, 4),
            'mean_speed': round(float(np.mean(v)) if v.size else 0.0, 4),
            'max_speed': round(float(np.max(v)) if v.size else 0.0, 4),
            'cum_risk_gt_time': round(cum_risk_time, 4),
            'cum_risk_gt_path': round(cum_risk_path, 4),
            'mean_risk_gt': round(float(np.mean(risk)) if risk.size else 0.0, 4),
            'peak_risk_gt': round(float(np.max(peak_layer)) if peak_layer.size else 0.0, 4),
            'high_risk_dwell_frac': round(high_dwell_frac, 4),
            'dwell_risk_03_s': round(dwell_risk_03_s, 3),
            'peak_risk_03': round(peak_risk_03, 4),
            'cum_risk_own_time': round(cum_own_time, 4),
            'mean_risk_own': round(float(np.mean(own)) if own.size else 0.0, 4),
            'risk_calibration_bias': round(float(risk_calibration_bias), 4),
            'collisions_static': self.collisions_static,
            'collisions_crowd': self.collisions_crowd,
            'collisions_total': self.collisions_static + self.collisions_crowd,
            'min_surface_clearance_crowd_m': round(float(finite_surf_crowd.min()) if finite_surf_crowd.size else float('inf'), 4),
            'mean_surface_clearance_crowd_m': round(float(finite_surf_crowd.mean()) if finite_surf_crowd.size else float('inf'), 4),
            'min_surface_clearance_obstacle_m': round(float(finite_surf_obs.min()) if finite_surf_obs.size else float('inf'), 4),
            'min_hazard_clearance_m': round(float(finite_haz.min()) if finite_haz.size else float('inf'), 4),
            'mean_hazard_clearance_m': round(float(finite_haz.mean()) if finite_haz.size else float('inf'), 4),
            'jerk_rms': round(jerk_rms, 4),
            'jerk_peak': round(jerk_peak, 4),
            'ang_jerk_rms': round(ang_jerk_rms, 4),
            'control_effort_lin': round(control_effort_lin, 4),
            'control_effort_ang': round(control_effort_ang, 4),
            'num_stops': num_stops,
            'final_xy_error_m': round(final_xy_err, 4),
            'n_samples': len(self.t),
        }

    def time_series(self) -> dict:
        return {
            't': np.asarray(self.t), 'x': np.asarray(self.x),
            'y': np.asarray(self.y), 'v': np.asarray(self.v),
            'w': np.asarray(self.w), 'risk_gt': np.asarray(self.risk),
            'elev': np.asarray(self.elev),
            'peak_layer': np.asarray(self.peak_layer),
            'risk_own': np.asarray(self.own_risk),
            'crowd_clearance': np.asarray(self.crowd_clear),
            'obstacle_clearance': np.asarray(self.obs_clear),
        }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    'run_id', 'method', 'goal', 'seed',
    'risk_density', 'overlap_cells', 'overlap_m2',
    'success', 'failure_reason', 'time_to_goal_s', 'path_length_m',
    'straight_line_m', 'spl', 'mean_speed', 'max_speed',
    'cum_risk_gt_time', 'cum_risk_gt_path', 'mean_risk_gt', 'peak_risk_gt',
    'high_risk_dwell_frac', 'dwell_risk_03_s', 'peak_risk_03',
    'cum_risk_own_time', 'mean_risk_own', 'risk_calibration_bias',
    'collisions_static', 'collisions_crowd', 'collisions_total',
    'min_surface_clearance_crowd_m', 'mean_surface_clearance_crowd_m',
    'min_surface_clearance_obstacle_m',
    'min_hazard_clearance_m', 'mean_hazard_clearance_m',
    'jerk_rms', 'jerk_peak', 'ang_jerk_rms',
    'control_effort_lin', 'control_effort_ang', 'num_stops',
    'final_xy_error_m', 'n_samples',
]


class MetricsNode(Node):

    def __init__(self) -> None:
        super().__init__('metrics_node')

        share = _maps_share_dir()
        self.declare_parameter('maps_dir', share)
        self.declare_parameter('scene_meta', os.path.join(share, 'scene_meta.json'))
        # Terrain-slope risk map that feeds /risk/elevation.
        self.declare_parameter('elevation_pgm',
                               os.path.join(share, 'terrain_slope.pgm'))
        # Run metadata (written into every row so the CSV is self-describing).
        self.declare_parameter('method', 'unknown')
        self.declare_parameter('seed', 0)
        self.declare_parameter('goal', '')           # optional label override
        self.declare_parameter('output_dir',
                               os.path.expanduser('~/sar_metrics'))
        self.declare_parameter('csv_name', 'runs.csv')
        self.declare_parameter('save_trajectory', True)
        # Geometry / thresholds.
        self.declare_parameter('robot_radius', 0.40)
        self.declare_parameter('inflation_radius', 0.40)
        self.declare_parameter('person_radius', 0.35)
        self.declare_parameter('goal_xy_tol', 0.45)
        self.declare_parameter('high_risk_thr', 0.50)
        # Run lifecycle. A run that hasn't reached the goal within
        # stuck_timeout_s is failed with reason 'stuck'. There is no separate
        # no-progress detector — this overall time cap is the only failure path.
        self.declare_parameter('stuck_timeout_s', 200.0)
        self.declare_parameter('use_cmd_vel', False)  # else twist from /odom

        maps_dir = self.get_parameter('maps_dir').value
        with open(self.get_parameter('scene_meta').value) as f:
            meta = json.load(f)
        self._meta = meta
        self._goals = {g['name']: g for g in meta.get('goals', [])}
        self.gt = GroundTruthField(
            maps_dir, meta, self.get_parameter('elevation_pgm').value)
        self.get_logger().info(
            f'GT risk field {self.gt.h}x{self.gt.w} built '
            f'(max risk {self.gt.risk.max():.2f}).')

        self._out_dir = self.get_parameter('output_dir').value
        os.makedirs(self._out_dir, exist_ok=True)
        self._csv_path = os.path.join(self._out_dir,
                                      self.get_parameter('csv_name').value)
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

        self._rec: Optional[RunRecorder] = None
        self._goal: Optional[Tuple[float, float, float]] = None
        self._start_pose: Optional[Tuple[float, float]] = None
        self._t0 = 0.0
        self._latest_cmd: Optional[Twist] = None
        self._crowd: List[Tuple[float, float]] = []
        self._own_risk_msg: Optional[OccupancyGrid] = None
        self._run_count = 0

        # /goal_pose: send_goal publishes VOLATILE, so subscribe VOLATILE too
        # (a TRANSIENT_LOCAL subscriber would be QoS-incompatible and get nothing).
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 30)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.create_subscription(MarkerArray, '/crowd_markers',
                                 self._on_crowd, 5)
        self.create_subscription(OccupancyGrid, '/risk_fused',
                                 self._on_fused, 5)

        self.get_logger().info(
            f"metrics_node armed — method='{self.get_parameter('method').value}', "
            f"writing {self._csv_path}")

    # -- helpers ------------------------------------------------------------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _goal_label(self) -> str:
        override = str(self.get_parameter('goal').value)
        if override:
            return override
        if self._goal is None:
            return ''
        gx, gy, _ = self._goal
        best, bestd = '', float('inf')
        for name, g in self._goals.items():
            d = math.hypot(g['x'] - gx, g['y'] - gy)
            if d < bestd:
                best, bestd = name, d
        return best if bestd < 1.0 else f'{gx:.1f}_{gy:.1f}'

    # -- callbacks ----------------------------------------------------------

    def _on_cmd(self, msg: Twist) -> None:
        self._latest_cmd = msg

    def _on_crowd(self, msg: MarkerArray) -> None:
        self._crowd = [(m.pose.position.x, m.pose.position.y)
                       for m in msg.markers if m.ns == 'crowd']

    def _on_fused(self, msg: OccupancyGrid) -> None:
        self._own_risk_msg = msg

    def _own_risk_at(self, x: float, y: float) -> float:
        m = self._own_risk_msg
        if m is None:
            return float('nan')
        c = int((x - m.info.origin.position.x) / m.info.resolution)
        r = int((y - m.info.origin.position.y) / m.info.resolution)
        if 0 <= r < m.info.height and 0 <= c < m.info.width:
            val = m.data[r * m.info.width + c]
            return val / 100.0 if val >= 0 else float('nan')
        return float('nan')

    def _on_goal(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
        gx, gy = msg.pose.position.x, msg.pose.position.y
        # send_goal republishes the SAME goal ~5x for reliability. Ignore repeats
        # of the goal we're already running on, otherwise each repeat would
        # preempt and re-arm the run (spamming preempted_by_new_goal rows).
        if self._goal is not None and \
                math.hypot(gx - self._goal[0], gy - self._goal[1]) < 0.1:
            return
        if self._rec is not None:        # a genuinely different goal arrived
            self._finish(False, 'preempted_by_new_goal')
        self._goal = (gx, gy, yaw)
        self._start_pose = None  # set on first odom
        self._rec = None         # created on first odom (need start pose)
        self._t0 = self._now()
        self.get_logger().info(f'Run armed for goal ({gx:.2f}, {gy:.2f}).')

    def _on_odom(self, msg: Odometry) -> None:
        if self._goal is None:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        t = self._now()

        if self._start_pose is None:
            self._start_pose = (x, y)
            self._rec = RunRecorder(
                self.gt, self._goal, self._start_pose,
                float(self.get_parameter('robot_radius').value),
                float(self.get_parameter('person_radius').value),
                float(self.get_parameter('high_risk_thr').value),
                float(self.get_parameter('inflation_radius').value))
            self._t0 = t

        # Velocity source: commanded /cmd_vel or ground-truth /odom twist.
        if self.get_parameter('use_cmd_vel').value and self._latest_cmd:
            v = self._latest_cmd.linear.x
            w = self._latest_cmd.angular.z
        else:
            lin = msg.twist.twist.linear
            v = math.hypot(lin.x, lin.y)
            w = msg.twist.twist.angular.z

        self._rec.add(t, x, y, v, w, self._own_risk_at(x, y), self._crowd)

        # End conditions.
        xy_tol = float(self.get_parameter('goal_xy_tol').value)
        if math.hypot(x - self._goal[0], y - self._goal[1]) <= xy_tol:
            self._finish(True, 'reached_goal')
            return
        # Failure is purely the overall time cap now (no no-progress detector):
        # if it hasn't reached the goal within stuck_timeout_s, mark it 'stuck'.
        if t - self._t0 > float(self.get_parameter('stuck_timeout_s').value):
            self._finish(False, 'stuck')
            return

    # -- finalisation -------------------------------------------------------

    def _finish(self, success: bool, reason: str) -> None:
        if self._rec is None:
            self._goal = None
            return
        xy_tol = float(self.get_parameter('goal_xy_tol').value)
        metrics = self._rec.finalize(success, reason, xy_tol)

        self._run_count += 1
        method = str(self.get_parameter('method').value)
        seed = int(self.get_parameter('seed').value)
        goal_label = self._goal_label()
        run_id = f'{method}_{goal_label}_s{seed}_{int(time.time())}'

        # Scene-level descriptors (constant within a run) from scene_meta, so
        # rows can be grouped by risk density / overlap in the analysis.
        scene = {'risk_density': self._meta.get('risk_density', 1.0),
                 'overlap_cells': self._meta.get('overlap_cells', ''),
                 'overlap_m2': self._meta.get('overlap_m2', '')}
        row = {'run_id': run_id, 'method': method, 'goal': goal_label,
               'seed': seed, **scene, **metrics}
        with open(self._csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        if self.get_parameter('save_trajectory').value:
            ts = self._rec.time_series()
            np.savez(os.path.join(self._out_dir, f'{run_id}.npz'),
                     method=method, goal=goal_label, seed=seed,
                     success=int(success), **ts)

        self.get_logger().info(
            f"[{reason}] {run_id}: success={metrics['success']} "
            f"t={metrics['time_to_goal_s']}s len={metrics['path_length_m']}m "
            f"cum_risk={metrics['cum_risk_gt_time']} "
            f"coll={metrics['collisions_total']} "
            f"min_haz={metrics['min_hazard_clearance_m']}m "
            f"jerk_rms={metrics['jerk_rms']}")

        self._rec = None
        self._goal = None
        self._start_pose = None


def _maps_share_dir() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('field_hospital_simulation'), 'maps')
    except Exception:
        return os.getcwd()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
