#!/usr/bin/env python3
"""sim_obstacle_node.py — ground-truth stand-in for the RTAB-Map obstacle grid
(/octomap_grid, /grid_prob_map) that risk_fusion_node and global_planner_node
consume.

Publishes the pre-baked ground-truth obstacle grid (obstacle_gt.npy) in the map
frame with octomap-style semantics (0 = free, 100 = occupied, -1 = unknown):

    publishes  /octomap_grid  nav_msgs/OccupancyGrid

Two reveal policies, matching how a field hospital is actually known:
  * structural obstacles (perimeter walls, ward tents) are part of a known
    floor plan -> revealed everywhere from the start (so global A* routes the
    corridors), controlled by `reveal_structural_global`.
  * movable clutter (cots, carts, standing staff) is only revealed within
    `reveal_radius` of the robot -> the local MPPI must react to it as it
    comes into range, exactly like the real perception stack.
"""

import array
import json
import math
import os

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class SimObstacleNode(Node):

    def __init__(self) -> None:
        super().__init__('sim_obstacle')

        self.declare_parameter('scene_meta', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('reveal_radius', 5.0)            # for movable clutter (m)
        self.declare_parameter('reveal_structural_global', True)
        self.declare_parameter('topic', '/octomap_grid')
        self.declare_parameter('publish_rate', 5.0)
        # Live walking crowd from sim_crowd_node, overlaid as movable obstacles
        # (shown only within the current sensor disk, so the local planner sees
        # people as they come into range rather than from a stale map).
        self.declare_parameter('crowd_topic', '/crowd_obstacles')

        meta_path = self.get_parameter('scene_meta').value
        if not meta_path or not os.path.isfile(meta_path):
            self.get_logger().fatal(f'scene_meta not found: {meta_path!r}')
            raise SystemExit(1)
        with open(meta_path) as f:
            meta = json.load(f)
        maps_dir = os.path.dirname(os.path.abspath(meta_path))

        self._res = float(meta['resolution'])
        self._ox  = float(meta['origin_x'])
        self._oy  = float(meta['origin_y'])
        self._occ = (np.load(os.path.join(maps_dir, meta['obstacle_gt'])) == 100)
        self._structural = np.load(os.path.join(maps_dir, meta['obstacle_structural'])).astype(bool)
        self._h, self._w = self._occ.shape
        self._revealed = np.zeros((self._h, self._w), dtype=bool)
        if self.get_parameter('reveal_structural_global').value:
            self._revealed |= self._structural

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._crowd_occ = None     # latest crowd footprints (bool, h x w)
        self.create_subscription(
            OccupancyGrid, self.get_parameter('crowd_topic').value,
            self._on_crowd, 1)

        topic = self.get_parameter('topic').value
        self.pub = self.create_publisher(OccupancyGrid, topic, 10)
        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'sim_obstacle: {self._w}x{self._h} @ {self._res} m/cell on {topic}; '
            f'structural_global={self.get_parameter("reveal_structural_global").value}, '
            f'reveal_radius={self.get_parameter("reveal_radius").value} m.')

    def _robot_cell(self):
        mf = self.get_parameter('map_frame').value
        rf = self.get_parameter('robot_frame').value
        try:
            tf = self._tf_buffer.lookup_transform(mf, rf, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'TF {mf}<-{rf} unavailable: {e}',
                                   throttle_duration_sec=5.0)
            return None
        x, y = tf.transform.translation.x, tf.transform.translation.y
        c = int(math.floor((x - self._ox) / self._res))
        r = int(math.floor((y - self._oy) / self._res))
        return r, c

    def _on_crowd(self, msg: OccupancyGrid) -> None:
        if msg.info.width == self._w and msg.info.height == self._h:
            arr = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape(
                self._h, self._w)
            self._crowd_occ = arr >= 100
        else:
            self.get_logger().warn(
                'crowd grid framing mismatch; ignoring.',
                throttle_duration_sec=10.0)

    def _reveal(self, r: int, c: int) -> None:
        radius = float(self.get_parameter('reveal_radius').value)
        if radius <= 0.0:
            self._revealed[:] = True
            return
        rad = int(math.ceil(radius / self._res))
        r0, r1 = max(0, r - rad), min(self._h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(self._w, c + rad + 1)
        if r0 >= r1 or c0 >= c1:
            return
        rr, cc = np.ogrid[r0:r1, c0:c1]
        disk = (rr - r) ** 2 + (cc - c) ** 2 <= rad * rad
        self._revealed[r0:r1, c0:c1] |= disk

    def _tick(self) -> None:
        rc = self._robot_cell()
        if rc is not None and 0 <= rc[0] < self._h and 0 <= rc[1] < self._w:
            self._reveal(rc[0], rc[1])

        grid = np.full((self._h, self._w), -1, dtype=np.int16)
        grid[self._revealed] = 0
        grid[self._revealed & self._occ] = 100

        # Overlay the live crowd as movable obstacles, but only within the
        # current sensor disk around the robot — people are detected as they
        # come into range, not remembered after they walk away.
        if self._crowd_occ is not None and rc is not None:
            radius = float(self.get_parameter('reveal_radius').value)
            if radius <= 0.0:
                grid[self._crowd_occ] = 100
            else:
                rad = int(math.ceil(radius / self._res))
                r0, r1 = max(0, rc[0] - rad), min(self._h, rc[0] + rad + 1)
                c0, c1 = max(0, rc[1] - rad), min(self._w, rc[1] + rad + 1)
                if r1 > r0 and c1 > c0:
                    rr, cc = np.ogrid[r0:r1, c0:c1]
                    disk = (rr - rc[0]) ** 2 + (cc - rc[1]) ** 2 <= rad * rad
                    seen = disk & self._crowd_occ[r0:r1, c0:c1]
                    grid[r0:r1, c0:c1][seen] = 100

        grid = grid.astype(np.int8)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('map_frame').value
        msg.info.resolution = self._res
        msg.info.width = self._w
        msg.info.height = self._h
        msg.info.origin.position.x = self._ox
        msg.info.origin.position.y = self._oy
        msg.info.origin.orientation.w = 1.0
        msg.data = array.array('b', grid.tobytes())
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
