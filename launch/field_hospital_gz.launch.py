#!/usr/bin/env python3
"""field_hospital_gz.launch.py — full sim on Gazebo Sim (Fortress / ogre2 PBR).

Starts Ignition Gazebo with the PBR world, bridges gz<->ROS, spawns the
wheelchair, and brings up the risk_stack.launch.py pipeline.

Prereq:  sudo apt install -y ros-humble-ros-gz
Run:     ros2 launch field_hospital_simulation field_hospital_gz.launch.py
         (headless:=true for no GUI)
Goal:    ros2 run field_hospital_simulation send_goal red_ward_patient
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('field_hospital_simulation')
    world = os.path.join(pkg, 'worlds', 'field_hospital_gz.sdf')
    xacro_file = os.path.join(pkg, 'urdf', 'wheelchair_gz.urdf.xacro')
    bridge_yaml = os.path.join(pkg, 'config', 'ros_gz_bridge.yaml')

    headless = LaunchConfiguration('headless')

    # gz resolves resource-root-relative texture URIs (media/...) against the
    # package share dir, and model:// prop meshes against the models/ dir.
    resource_env = AppendEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH', value=pkg)
    models_env = AppendEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH', value=os.path.join(pkg, 'models'))

    # Run Ignition Gazebo (Fortress). -r runs immediately; -s = server only.
    gz_gui = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-v', '3', world],
        output='screen', condition=UnlessCondition(headless))
    gz_headless = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', '-v', '3', world],
        output='screen', condition=IfCondition(headless))

    robot_desc = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])

    # Spawn pose is parametrised so a run can start the chair outside or inside
    # a tent (the tent->tent mission legs). /odom is ground-truth world pose and
    # map->odom is identity, so relocating the spawn needs no other change.
    #
    # `leg:=<name>` is a convenience: it looks up that leg's `start` pose in
    # scene_meta.json and spawns there, so you don't have to keep spawn_x/y/yaw
    # in sync with the leg by hand (the easiest way to accidentally start a leg
    # in the wrong place). Empty `leg` (default) falls back to spawn_x/y/yaw.
    def _spawn_node(context, *_a, **_k):
        leg = LaunchConfiguration('leg').perform(context).strip()
        sx = LaunchConfiguration('spawn_x').perform(context)
        sy = LaunchConfiguration('spawn_y').perform(context)
        syaw = LaunchConfiguration('spawn_yaw').perform(context)
        if leg:
            import json
            try:
                with open(os.path.join(pkg, 'maps', 'scene_meta.json')) as f:
                    goals = {g['name']: g for g in json.load(f).get('goals', [])}
            except OSError:
                goals = {}
            if leg in goals:
                st = goals[leg].get('start', {'x': -2.0, 'y': 0.0, 'yaw': 0.0})
                sx, sy, syaw = str(st['x']), str(st['y']), str(st.get('yaw', 0.0))
                print(f"[field_hospital_gz] leg='{leg}' -> spawn "
                      f"({sx}, {sy}, yaw={syaw})")
            else:
                print(f"[field_hospital_gz] WARNING: unknown leg '{leg}'; using "
                      f"spawn_x/y/yaw=({sx}, {sy}, {syaw}). Known: {sorted(goals)}")
        return [Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=['-topic', 'robot_description', '-name', 'wheelchair',
                       '-x', sx, '-y', sy, '-z', '0.20', '-Y', syaw])]

    # Default the GUI "Follow" camera onto the wheelchair so it tracks the
    # robot inside tents instead of staying outside looking down (Fortress's
    # CameraTracking GUI plugin has no static SDF config for this — it's
    # service-only — so we call /gui/follow + /gui/follow/offset once the GUI
    # is up). Offset is in the robot's local frame: 1.5 m behind, 1 m up,
    # comfortably under the 2.78 m tent ridge. Retries for ~10 s in case the
    # GUI is still loading the (large) world; still adjustable live afterward.
    camera_follow = TimerAction(
        period=3.0,
        actions=[ExecuteProcess(
            cmd=['bash', '-c', (
                'for i in $(seq 1 10); do '
                'ign service -s /gui/follow '
                '--reqtype ignition.msgs.StringMsg --reptype ignition.msgs.Boolean '
                '--timeout 1000 --req \'data: "wheelchair"\' 2>/dev/null && '
                'ign service -s /gui/follow/offset '
                '--reqtype ignition.msgs.Vector3d --reptype ignition.msgs.Boolean '
                '--timeout 1000 --req \'x: -3.5, y: 0.0, z: 1.7\' 2>/dev/null '
                '&& break; sleep 1; done'
            )],
            output='screen')],
        condition=UnlessCondition(headless))

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='ros_gz_bridge', output='screen',
        parameters=[{'config_file': bridge_yaml, 'use_sim_time': True}])

    map_to_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': True}])

    # Reliable odom->base_link TF from the bridged /odom (replaces the flaky
    # gz Pose_V -> /tf bridge that made the planner lose the pose and spin).
    odom_tf = Node(
        package='field_hospital_simulation', executable='odom_tf',
        name='odom_tf', output='screen',
        parameters=[{'use_sim_time': True}])

    risk_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'risk_stack.launch.py')),
        launch_arguments={
            'known_floorplan': LaunchConfiguration('known_floorplan'),
            'terrain_reveal_radius': LaunchConfiguration('terrain_reveal_radius'),
            'obstacle_reveal_radius': LaunchConfiguration('obstacle_reveal_radius'),
            'rviz': LaunchConfiguration('rviz'),
            'fusion_method': LaunchConfiguration('fusion_method'),
            'use_risk_map': LaunchConfiguration('use_risk_map'),
            'mppi_seed': LaunchConfiguration('mppi_seed'),
            'crowd_phase_offset': LaunchConfiguration('crowd_phase_offset'),
            'local_mu_blocked': LaunchConfiguration('local_mu_blocked'),
            'metrics': LaunchConfiguration('metrics'),
            'metrics_goal': LaunchConfiguration('metrics_goal'),
            'metrics_out': LaunchConfiguration('metrics_out'),
        }.items())

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('known_floorplan', default_value='true'),
        DeclareLaunchArgument('terrain_reveal_radius', default_value='4.0'),
        DeclareLaunchArgument('obstacle_reveal_radius', default_value='5.0'),
        # Forwarded to risk_stack for the Monte-Carlo fusion study.
        DeclareLaunchArgument('fusion_method', default_value='independent',
                              choices=['independent', 'max', 'logodds']),
        DeclareLaunchArgument('use_risk_map', default_value='true',
                              choices=['true', 'false']),
        DeclareLaunchArgument('mppi_seed', default_value='-1'),
        DeclareLaunchArgument('crowd_phase_offset', default_value='0.0'),
        DeclareLaunchArgument('local_mu_blocked', default_value='0.8'),
        DeclareLaunchArgument('metrics', default_value='false'),
        DeclareLaunchArgument('metrics_goal', default_value=''),
        DeclareLaunchArgument('metrics_out',
                              default_value=os.path.expanduser('~/sar_metrics')),
        # Robot spawn pose (default = outside the compound, as before).
        # `leg:=<name>` overrides these with that leg's start pose from
        # scene_meta; empty (default) uses spawn_x/y/yaw directly.
        DeclareLaunchArgument('leg', default_value=''),
        DeclareLaunchArgument('spawn_x', default_value='-2.0'),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_yaw', default_value='0.0'),
        resource_env,
        models_env,
        gz_gui,
        gz_headless,
        rsp,
        OpaqueFunction(function=_spawn_node),
        camera_follow,
        bridge,
        map_to_odom,
        odom_tf,
        risk_stack,
    ])
