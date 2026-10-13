#!/usr/bin/env bash
# kill_sim.sh — nuke every process from the field-hospital sim so stale stacks
# don't pile up between runs (duplicate nodes cause the robot to spin, the map
# to show an old layout, etc.). Run this before each fresh launch.
set -u

PATTERNS=(
  'ros2 launch'
  'field_hospital'
  'risk_stack'
  'sim_obstacle' 'sim_terrain' 'sim_crowd' 'odom_tf' 'send_goal'
  'field_hospital_simulation/metrics' 'metrics_node'
  'local_planner' 'global_planner' 'risk_fusion' 'elevation_map'
  'parameter_bridge' 'ros_gz'
  'robot_state_publisher' 'static_transform_publisher'
  'rviz2'
  'ign gazebo' 'gzserver' 'gzclient' 'ruby.*ign' 'gz sim'
)

for p in "${PATTERNS[@]}"; do
  pkill -9 -f "$p" 2>/dev/null
done
sleep 2

# Report anything still alive so you know if a terminal needs closing.
LEFT=$(pgrep -af 'sim_obstacle|local_planner|ign gazebo|parameter_bridge|risk_fusion' || true)
if [ -n "$LEFT" ]; then
  echo "STILL RUNNING (close that terminal manually):"
  echo "$LEFT"
else
  echo "All sim processes cleared."
fi
