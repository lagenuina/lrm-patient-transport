#!/usr/bin/env python3
"""send_goal — publish a /goal_pose in the map frame.

  ros2 run field_hospital_simulation send_goal red_ward_patient   # a named scene goal
  ros2 run field_hospital_simulation send_goal 22.5 0.0           # raw x y [yaw]

Named goals come from the installed scene_meta.json.
"""

import math
import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def _load_goals():
    import json
    meta = os.path.join(get_package_share_directory('field_hospital_simulation'),
                        'maps', 'scene_meta.json')
    with open(meta) as f:
        return {g['name']: g for g in json.load(f)['goals']}


def main():
    args = sys.argv[1:]
    goals = _load_goals()
    if len(args) == 1 and args[0] in goals:
        g = goals[args[0]]
        x, y, yaw = g['x'], g['y'], g.get('yaw', 0.0)
    elif len(args) >= 2:
        x, y = float(args[0]), float(args[1])
        yaw = float(args[2]) if len(args) > 2 else 0.0
    else:
        print(f'Usage: send_goal.py <name|x y [yaw]>\nNamed goals: {list(goals)}')
        return

    rclpy.init()
    node = Node('send_goal')
    pub = node.create_publisher(PoseStamped, '/goal_pose', 1)
    msg = PoseStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)

    # /goal_pose is VOLATILE, so a message sent before pub<->sub discovery
    # completes is dropped (no late delivery). The old fixed 5x/0.75s burst could
    # finish before a subscriber matched on a busy first launch, leaving the
    # planner — and especially metrics_node — never receiving the goal (symptom:
    # "no row written (timeout)"). Instead publish at ~10 Hz until at least one
    # subscriber is discovered, then keep publishing a short grace burst so every
    # late-matching subscriber (planner AND metrics) gets it; cap the total wait.
    import time
    MAX_WAIT, GRACE = 15.0, 2.0
    t_start = time.monotonic()
    t_first_sub = None
    while True:
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.1)
        now = time.monotonic()
        if pub.get_subscription_count() > 0 and t_first_sub is None:
            t_first_sub = now
        if t_first_sub is not None and now - t_first_sub >= GRACE:
            break
        if now - t_start >= MAX_WAIT:
            node.get_logger().warn(
                f'goal published but no subscriber appeared in {MAX_WAIT:.0f}s.')
            break
    node.get_logger().info(
        f'Sent goal ({x}, {y}, yaw={yaw}) to '
        f'{pub.get_subscription_count()} subscriber(s).')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
