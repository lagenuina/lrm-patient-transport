#!/usr/bin/env python3
"""sim_terrain_node.py — ground-truth stand-in for terrain_map_node (SegFormer).

Instead of segmenting RGB, it reveals the pre-baked ground-truth terrain-cost
grid (terrain_gt.npy, written by field_hospital_scene) within sensor range of
the robot and republishes it on the SAME topic / frame / units as the real
node, so risk_fusion_node and the planners run completely unmodified:

    publishes  /risk/terrain   nav_msgs/OccupancyGrid  (0..100 cost, -1 unknown)
    frame      map
    needs      TF map -> robot_frame

Cells the robot has never come within `reveal_radius` of stay unknown (-1), so
the planner still discovers terrain as it drives — matching the incremental,
robot-centred behaviour of the real perception node.
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


class SimTerrainNode(Node):

    def __init__(self) -> None:
        super().__init__('sim_terrain')

        self.declare_parameter('scene_meta', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('reveal_radius', 4.0)   # sensor range (m); <=0 = full GT
        self.declare_parameter('publish_rate', 5.0)

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
        self._cost = np.load(os.path.join(maps_dir, meta['terrain_gt'])).astype(np.int8)
        self._h, self._w = self._cost.shape
        self._revealed = np.zeros((self._h, self._w), dtype=bool)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.pub = self.create_publisher(OccupancyGrid, '/risk/terrain', 10)
        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'sim_terrain: {self._w}x{self._h} @ {self._res} m/cell, '
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

        grid = np.where(self._revealed, self._cost.astype(np.int16), -1).astype(np.int8)
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
    node = SimTerrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
