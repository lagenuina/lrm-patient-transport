#!/usr/bin/env python3
"""risk_stack.launch.py — the risk + planning stack, included by
field_hospital_gz.launch.py.

Brings up everything that is NOT the simulator: the GT perception stand-ins,
the static risk layer, fusion, and the global/local planners (+ RViz).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context, *args, **kwargs):
    sim_pkg = get_package_share_directory('field_hospital_simulation')
    maps = os.path.join(sim_pkg, 'maps')
    scene_meta = os.path.join(maps, 'scene_meta.json')

    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    pgm_yaml = os.path.join(maps, 'terrain_slope.yaml')

    # 'No risk map' ablation: when use_risk_map is false the local planner
    # ignores the fused risk (see local_planner_node). The fusion rule is then
    # irrelevant, so label the run 'norisk' for metrics instead of the (unused)
    # fusion_method; otherwise the metrics label is the fusion rule itself.
    use_risk_map = cfg('use_risk_map').lower() == 'true'
    method_label = cfg('fusion_method') if use_risk_map else 'norisk'

    cell_size = 0.1
    use_sim = {'use_sim_time': True}

    elevation = Node(
        package='risk_maps', executable='elevation_map',
        name='elevation_map_node', output='screen',
        parameters=[{'map_yaml': pgm_yaml, 'topic': 'risk/elevation',
                     'frame_id': 'map', 'publish_rate': 1.0,
                     # origin_x/origin_y left unset (NaN) so the node reads the
                     # origin from the generated map YAML — keeps the map extent
                     # defined in ONE place (field_hospital_scene X0/Y0) instead
                     # of duplicating it here where it can drift out of sync.
                     'cell_size': cell_size,
                     **use_sim}])

    sim_terrain = Node(
        package='field_hospital_simulation', executable='sim_terrain',
        name='sim_terrain', output='screen',
        parameters=[{'scene_meta': scene_meta, 'map_frame': 'map',
                     'robot_frame': 'base_link',
                     'reveal_radius': float(cfg('terrain_reveal_radius')),
                     'publish_rate': 5.0, **use_sim}])

    sim_obstacle = Node(
        package='field_hospital_simulation', executable='sim_obstacle',
        name='sim_obstacle', output='screen',
        parameters=[{'scene_meta': scene_meta, 'map_frame': 'map',
                     'robot_frame': 'base_link', 'topic': '/octomap_grid',
                     'reveal_radius': float(cfg('obstacle_reveal_radius')),
                     'reveal_structural_global': cfg('known_floorplan').lower() == 'true',
                     'publish_rate': 5.0, **use_sim}])

    sim_crowd = Node(
        package='field_hospital_simulation', executable='sim_crowd',
        name='sim_crowd', output='screen',
        parameters=[{'scene_meta': scene_meta, 'map_frame': 'map',
                     # Walking speed is baked into the waypoint times at scene
                     # generation (field_hospital_scene.CROWD_SPEED), not here.
                     'publish_rate': 10.0,
                     'phase_offset': float(cfg('crowd_phase_offset')),
                     'crowd_topic': '/crowd_obstacles', **use_sim}])

    fusion = Node(
        package='risk_maps', executable='risk_fusion',
        name='risk_fusion', output='screen',
        parameters=[{'radius': 8.0, 'cell_size': cell_size, 'publish_rate': 20.0,
                     'map_frame': 'map', 'robot_frame': 'base_link',
                     'obstacle_topic': '/octomap_grid',
                     'obstacle_inflation_radius': 0.25,
                     'obstacle_seed_threshold': 0.5,
                     'fusion_method': cfg('fusion_method'), **use_sim}])

    global_planner = Node(
        package='risk_maps', executable='global_planner',
        name='global_planner', output='screen',
        parameters=[{'mu_blocked': 0.8, 'planning_resolution': 0.05,
                     'inflation_radius': 0.6,
                     'direct_distance': 3.0, 'map_frame': 'map',
                     'robot_frame': 'base_link', 'obstacle_topic': '/octomap_grid',
                     **use_sim}])

    local_planner = Node(
        package='risk_maps', executable='local_planner',
        name='local_planner', output='screen',
        parameters=[{'mu_blocked': float(cfg('local_mu_blocked')),
                     'seed': int(cfg('mppi_seed')),
                     'publish_rate': 20.0, 'v_min': -0.2, 'v_max': 0.5,
                     'w_max': 1.0, 'map_frame': 'map', 'robot_frame': 'base_link',
                     'use_luci': False, 'use_risk_map': use_risk_map,
                     **use_sim}])

    risk_viz = Node(
        package='risk_maps', executable='risk_map_viz',
        name='risk_map_viz', output='screen',
        parameters=[{'input_topic': '/risk_fused',
                     'output_topic': '/risk_fused_viz',
                     'min_risk': 0.04, 'blur_sigma': 0.5,
                     'alpha_min': 0.0, 'alpha_max': 0.92,
                     **use_sim}])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', os.path.join(sim_pkg, 'rviz', 'field_hospital.rviz')],
        condition=IfCondition(cfg('rviz')), parameters=[use_sim], output='screen')

    # Optional per-run evaluation logger (Monte-Carlo fusion study).
    elev_pgm = os.path.join(maps, 'terrain_slope.pgm')
    metrics = Node(
        package='field_hospital_simulation', executable='metrics',
        name='metrics_node', output='screen',
        condition=IfCondition(cfg('metrics')),
        parameters=[{'scene_meta': scene_meta, 'maps_dir': maps,
                     'elevation_pgm': elev_pgm,
                     'method': method_label,
                     'seed': int(cfg('mppi_seed')),
                     'goal': cfg('metrics_goal'),
                     'output_dir': cfg('metrics_out'), **use_sim}])

    nodes = [elevation, sim_terrain, sim_obstacle, sim_crowd, fusion,
             global_planner, local_planner, risk_viz, rviz, metrics]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('known_floorplan', default_value='true'),
        DeclareLaunchArgument('terrain_reveal_radius', default_value='4.0'),
        DeclareLaunchArgument('obstacle_reveal_radius', default_value='5.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        # --- Monte-Carlo fusion study knobs (defaults preserve normal use) ---
        DeclareLaunchArgument('fusion_method', default_value='independent',
                              choices=['independent', 'max', 'logodds']),
        # false => local planner ignores the risk map ('norisk' ablation).
        DeclareLaunchArgument('use_risk_map', default_value='true',
                              choices=['true', 'false']),
        DeclareLaunchArgument('mppi_seed', default_value='-1'),
        DeclareLaunchArgument('crowd_phase_offset', default_value='0.0'),
        DeclareLaunchArgument('local_mu_blocked', default_value='0.8'),
        DeclareLaunchArgument('metrics', default_value='false'),
        DeclareLaunchArgument('metrics_goal', default_value=''),
        DeclareLaunchArgument('metrics_out',
                              default_value=os.path.expanduser('~/sar_metrics')),
        OpaqueFunction(function=_nodes),
    ])
