#!/usr/bin/env python3
"""odom_tf_node.py — broadcast the odom->base_link TF from the bridged /odom.

In the Gazebo Sim (Fortress) bringup, relying on the gz DiffDrive Pose_V -> /tf
bridge proved flaky: the planner intermittently lost map->base_link, so it kept
stopping and then slamming max turn (the robot span/circled). nav_msgs/Odometry
bridges reliably, so this node turns it straight into a TF, stamped from the
message itself (i.e. sim time) so it never goes stale relative to the listener.

    subscribes  /odom        nav_msgs/Odometry  (from ros_gz_bridge)
    broadcasts  odom -> base_link  (frames taken from the odom header)
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomTf(Node):
    def __init__(self) -> None:
        super().__init__('odom_tf')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self._br = TransformBroadcaster(self)
        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self._cb, 20)
        self.get_logger().info('odom_tf: /odom -> TF odom->base_link')

    def _cb(self, msg: Odometry) -> None:
        t = TransformStamped()
        # Use the message's own stamp (sim time) so the TF is never "in the
        # future/past" relative to a sim-time listener.
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id or self.get_parameter('odom_frame').value
        t.child_frame_id = msg.child_frame_id or self.get_parameter('base_frame').value
        p = msg.pose.pose
        t.transform.translation.x = p.position.x
        t.transform.translation.y = p.position.y
        t.transform.translation.z = p.position.z
        t.transform.rotation = p.orientation
        self._br.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
