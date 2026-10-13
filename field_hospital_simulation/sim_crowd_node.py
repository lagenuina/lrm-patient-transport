#!/usr/bin/env python3
"""sim_crowd_node.py — make the walking crowd real, dynamic obstacles for the
planner (the Gazebo *visual* is the walking actors; this node makes the planner
react to them).

The actors in the world walk looping waypoint trajectories from
scene_meta['crowd']. This node replays the SAME waypoints against sim time, so
its idea of where each person is matches the actor on screen without any
feedback or drift. From that it publishes:

    /crowd_obstacles  (nav_msgs/OccupancyGrid) people footprints; merged into
                      /octomap_grid by sim_obstacle_node (revealed within sensor
                      range, like any movable obstacle) so the local planner
                      dodges them
    /crowd_markers    (visualization_msgs/MarkerArray) cylinders for RViz

Position at sim time t is a looping linear interpolation of the waypoints
(period = last waypoint time), matching how the actor follows its trajectory.
"""

import array
import json
import math
import os

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class SimCrowdNode(Node):

    def __init__(self) -> None:
        super().__init__('sim_crowd')

        self.declare_parameter('scene_meta', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('person_radius', 0.35)    # obstacle footprint (m)
        # Shift the crowd's looping phase by this many seconds. With the sim
        # relaunched per run (so the clock restarts at 0), this makes the crowd
        # configuration reproducible and lets each Monte-Carlo seed be a
        # distinct-but-repeatable scenario shared across the fusion methods.
        self.declare_parameter('phase_offset', 0.0)
        self.declare_parameter('crowd_topic', '/crowd_obstacles')
        self.declare_parameter('markers_topic', '/crowd_markers')

        meta_path = self.get_parameter('scene_meta').value
        if not meta_path or not os.path.isfile(meta_path):
            self.get_logger().fatal(f'scene_meta not found: {meta_path!r}')
            raise SystemExit(1)
        with open(meta_path) as f:
            meta = json.load(f)

        self._res = float(meta['resolution'])
        self._ox  = float(meta['origin_x'])
        self._oy  = float(meta['origin_y'])
        self._w   = int(meta['width'])
        self._h   = int(meta['height'])
        self._crowd = meta.get('crowd', [])
        if not self._crowd:
            self.get_logger().warn('scene_meta has no crowd; nothing to do.')

        self.pub_grid = self.create_publisher(
            OccupancyGrid, self.get_parameter('crowd_topic').value, 1)
        self.pub_markers = self.create_publisher(
            MarkerArray, self.get_parameter('markers_topic').value, 1)

        rate = float(self.get_parameter('publish_rate').value)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'sim_crowd: tracking {len(self._crowd)} walking actors @ '
            f'{rate:.0f} Hz on a {self._w}x{self._h} grid.')

    @staticmethod
    def _pose_at(wps, t):
        """Looping linear interpolation of [t, x, y, yaw] waypoints."""
        period = wps[-1][0]
        if period <= 0.0:
            return wps[0][1], wps[0][2], wps[0][3]
        tt = t % period
        for i in range(len(wps) - 1):
            t0, t1 = wps[i][0], wps[i + 1][0]
            if t0 <= tt <= t1 and t1 > t0:
                f = (tt - t0) / (t1 - t0)
                return (wps[i][1] + f * (wps[i + 1][1] - wps[i][1]),
                        wps[i][2] + f * (wps[i + 1][2] - wps[i][2]),
                        wps[i][3])
        return wps[-1][1], wps[-1][2], wps[-1][3]

    def _positions(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        t += float(self.get_parameter('phase_offset').value)
        out = []
        for c in self._crowd:
            wps = c.get('waypoints') or [[0.0, c['x'], c['y'], c.get('yaw', 0.0)]]
            out.append(self._pose_at(wps, t))
        return out

    def _tick(self) -> None:
        if not self._crowd:
            return
        positions = self._positions()
        self._publish_grid(positions)
        self._publish_markers(positions)

    def _publish_grid(self, positions) -> None:
        grid = np.zeros((self._h, self._w), dtype=np.int8)
        rad = max(1, int(round(
            float(self.get_parameter('person_radius').value) / self._res)))
        for (x, y, _yaw) in positions:
            c = int((x - self._ox) / self._res)
            r = int((y - self._oy) / self._res)
            r0, r1 = max(0, r - rad), min(self._h, r + rad + 1)
            c0, c1 = max(0, c - rad), min(self._w, c + rad + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            rr, cc = np.ogrid[r0:r1, c0:c1]
            disk = (rr - r) ** 2 + (cc - c) ** 2 <= rad * rad
            grid[r0:r1, c0:c1][disk] = 100

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
        self.pub_grid.publish(msg)

    def _publish_markers(self, positions) -> None:
        now = self.get_clock().now().to_msg()
        frame = self.get_parameter('map_frame').value
        arr = MarkerArray()
        for i, (x, y, _yaw) in enumerate(positions):
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame
            m.ns = 'crowd'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.85
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.6
            m.scale.z = 1.7
            m.color = ColorRGBA(r=0.20, g=0.35, b=0.75, a=0.9)
            arr.markers.append(m)
        self.pub_markers.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimCrowdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
