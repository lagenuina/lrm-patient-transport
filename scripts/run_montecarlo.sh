#!/usr/bin/env bash
# run_montecarlo.sh — paired Monte-Carlo sweep over fusion methods.
#
# For every (seed, goal) the three fusion methods run under IDENTICAL conditions
# so the fusion rule is the only difference:
#   * the sim is relaunched per run, so the clock restarts at 0 -> the crowd
#     starts at the same phase and the robot at the same pose every time;
#   * the MPPI sampler is seeded (mppi_seed) so the noise draws are reproducible;
#   * the crowd phase is offset by the seed (crowd_phase_offset) so each seed is
#     a distinct-but-repeatable scenario, shared across the three methods.
#
# Teardown uses the repo's kill_sim.sh (robust pkill patterns) — process-group
# kills don't reliably catch the detached gz server. Each run is torn down
# fully before the next launches so stale nodes never pile up.
#
# Results append to $OUT/runs.csv (+ per-run .npz). Aggregate afterwards with
#   python3 aggregate_metrics.py $OUT/runs.csv --baseline independent
#
# Usage:  ./run_montecarlo.sh [N_SEEDS [START_SEED]]
#   N_SEEDS    total number of seeds to run (default 20)
#   START_SEED first seed index (default 0); use to resume a partial sweep,
#              e.g. `./run_montecarlo.sh 20 3` resumes from seed 3.
# Source your ROS + workspace install first (e.g. `source install/setup.bash`).
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
KILL_SIM="$HERE/kill_sim.sh"

# Fusion rules plus a 'norisk' ablation baseline: the local planner ignores the
# risk map (use_risk_map:=false) but still avoids hard obstacles, so we can see
# what the risk map buys vs. just driving to the goal around walls.
METHODS=(independent max logodds norisk)
# Goals are sampled per seed (one per scenario type) — see the rng snippet
# inside the seed loop below. All 20 named goals live in scene_meta.json.
N_SEEDS="${1:-20}"
START_SEED="${2:-0}"
# Risk-density sweep: scales terrain + slope patch counts per scene (see
# generate_scene --risk-density). Each level multiplies the total run count, so
# trim N_SEEDS when sweeping several. Override with e.g.
#   RISK_DENSITIES="1.0" ./run_montecarlo.sh        # single (current) density
DENSITIES=(${RISK_DENSITIES:-0.5 1.0 1.5})

OUT="${SAR_METRICS_OUT:-$HOME/sar_metrics}"
LOCAL_MU_BLOCKED=0.8        # hard wall at mu>=0.8 so the chair avoids tent
                            # walls/beds instead of wedging (matches global planner)
READY_TIMEOUT=150           # s to wait for the stack to come up (gz PBR load is slow)
RUN_TIMEOUT=330             # s hard cap per run (> metrics' stuck_timeout_s=300
                            # so the node writes its 'stuck' row before we give up)
SETTLE=5                    # s after readiness before sending the goal

# VIZ=1 ./run_montecarlo.sh  -> show the Gazebo GUI + RViz for EVERY run (watch
# the whole sweep). Default 0 = headless, no RViz (faster). Because the sim is
# relaunched per run (the clock restarts at 0, which a single persistent RViz
# can't follow cleanly), each run opens its own short-lived GUI+RViz that closes
# at teardown and reopens for the next run. That adds GUI load time per run, so
# a full sweep is slower with VIZ=1 — but it does run every condition.
VIZ="${VIZ:-0}"
if [[ "$VIZ" == 1 ]]; then HEADLESS=false; RVIZ=true; else HEADLESS=true; RVIZ=false; fi

mkdir -p "$OUT"
CSV="$OUT/runs.csv"

SIM_PROCS='ign gazebo|install/sar_risk|parameter_bridge|robot_state_publisher|ros_gz_sim'

teardown() {
  bash "$KILL_SIM" >/dev/null 2>&1
  # kill_sim's pkill is async + gz can be slow to die; poll until the graph is
  # actually empty before the next launch, or a stale node poisons DDS discovery
  # (symptom: `create` stuck "Waiting messages on topic [robot_description]").
  for _ in $(seq 1 15); do
    [ "$(pgrep -f "$SIM_PROCS" 2>/dev/null | grep -vc $$)" -eq 0 ] && break
    pkill -9 -f 'ign gazebo'        2>/dev/null
    pkill -9 -f 'install/sar_risk'  2>/dev/null
    pkill -9 -f 'parameter_bridge'  2>/dev/null
    pkill -9 -f 'robot_state_publisher' 2>/dev/null
    sleep 1
  done
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  sleep 2
}
trap teardown EXIT INT TERM

wait_for_odom() {  # block until /odom is publishing or READY_TIMEOUT
  local t=0
  until ros2 topic list 2>/dev/null | grep -q '^/odom$'; do
    sleep 3; t=$((t+3)); [[ $t -ge $READY_TIMEOUT ]] && return 1
  done
  timeout 20 ros2 topic echo /odom --once >/dev/null 2>&1
}

# Count DATA rows (exclude the CSV header). metrics_node writes the header when
# it first creates the file, so counting raw lines would make that header look
# like a finished run on the very first run (fresh CSV) -> the harness tears the
# first run down before it does anything. Subtracting the header fixes that.
rows() {
  if [[ -f "$CSV" ]]; then
    local n; n=$(wc -l < "$CSV"); echo $(( n > 0 ? n - 1 : 0 ))
  else
    echo 0
  fi
}

print_last_metrics() {  # pretty-print the most recent runs.csv row to the terminal
  python3 - "$CSV" <<'PY'
import csv, sys, math
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    sys.exit()
r = rows[-1]
order = ['success', 'failure_reason', 'time_to_goal_s', 'path_length_m',
         'straight_line_m', 'path_efficiency', 'spl', 'mean_speed', 'max_speed',
         'cum_risk_gt_time', 'cum_risk_gt_path', 'mean_risk_gt', 'peak_risk_gt',
         'high_risk_dwell_frac', 'elevation_exposure', 'mean_risk_own',
         'collisions_static', 'collisions_crowd', 'collisions_total',
         'min_clearance_crowd_m', 'min_hazard_clearance_m', 'mean_hazard_clearance_m',
         'jerk_rms', 'jerk_peak', 'ang_jerk_rms', 'control_effort_lin', 'num_stops',
         'final_xy_error_m', 'n_samples']
w = max(len(k) for k in order)
print(f"  ---- metrics: {r.get('method')} / {r.get('goal')} / seed {r.get('seed')} ----")
for k in order:
    if k in r:
        print(f"    {k:<{w}} : {r[k]}")
PY
}

# Install share dir the launched nodes read maps/worlds from. We regenerate the
# scene straight into it so a colcon rebuild isn't needed between seeds.
SHARE=$(python3 -c "from ament_index_python.packages import get_package_share_directory as g; print(g('field_hospital_simulation'))")
GEN="$HERE/generate_scene.py"   # source copy — imports the updated scene module

regen_scene() {  # $1 = seed, $2 = risk_density; rebuild terrain + world in-place
  # Pin --crowd-seed to the run seed so the crowd's random walk is reproducible
  # and identical across the methods for this (density, seed) cell (paired
  # comparison). Interactive generate_scene.py runs omit it => fresh random crowd.
  python3 "$GEN" --seed "$1" --crowd-seed "$1" --risk-density "$2" \
    --maps-dir "$SHARE/maps" --world-dir "$SHARE/worlds" >/dev/null 2>&1
}

start_pose() {  # $1 = goal name; echoes "x y yaw" of that leg's spawn pose
  python3 - "$SHARE/maps/scene_meta.json" "$1" <<'PY'
import json, sys
meta, name = sys.argv[1], sys.argv[2]
g = {x['name']: x for x in json.load(open(meta))['goals']}[name]
s = g.get('start', {'x': -2.0, 'y': 0.0, 'yaw': 0.0})
print(s['x'], s['y'], s['yaw'])
PY
}

teardown   # clean slate before we start

# Pre-flight: generate seed=0 at the first density so the maps are valid
# before the first launch and any errors surface immediately.
echo "--- pre-flight: generating scene seed=0 density=${DENSITIES[0]} ---"
python3 "$GEN" --seed 0 --crowd-seed 0 --risk-density "${DENSITIES[0]}" \
  --maps-dir "$SHARE/maps" --world-dir "$SHARE/worlds"
if [[ $? -ne 0 ]]; then
  echo "!! scene generation failed — check generate_scene.py and dependencies"; exit 1
fi
echo "--- pre-flight OK ---"

for density in "${DENSITIES[@]}"; do
for seed in $(seq "$START_SEED" $((START_SEED + N_SEEDS - 1))); do
  echo "--- regenerating scene for density=$density seed=$seed ---"
  if ! regen_scene "$seed" "$density"; then
    echo "  !! scene generation failed (density=$density seed=$seed) — skipping"; continue
  fi

  # Pick one goal per scenario type, deterministically from the seed.
  # random.Random(seed) gives the same sequence for the same seed on any
  # platform/Python version — paired runs across methods are identical.
  mapfile -t GOALS < <(python3 - "$seed" <<'PY'
import random, sys
tents = ['nw', 'ne', 'se', 'sw']
rng = random.Random(int(sys.argv[1]))
intake_dst = rng.choice(tents)
intra_src  = rng.choice(tents)
intra_dst  = rng.choice([t for t in tents if t != intra_src])
egress_src = rng.choice(tents)
print(f"intake_to_{intake_dst}")
print(f"intra_{intra_src}_to_{intra_dst}")
print(f"egress_from_{egress_src}")
PY
  )
  echo "  goals: ${GOALS[*]}"

  for goal in "${GOALS[@]}"; do
    read -r sx sy syaw <<< "$(start_pose "$goal")"
    for method in "${METHODS[@]}"; do
      echo "=== density=$density seed=$seed goal=$goal method=$method (spawn $sx,$sy,$syaw) ==="

      # 'norisk' baseline: keep a valid fusion_method (the launch validates the
      # choice) but tell the local planner to ignore the risk map. The launch
      # then labels the metrics row 'norisk'. The fusion rules use the risk map.
      if [[ "$method" == norisk ]]; then
        fmethod=independent; use_risk=false
      else
        fmethod=$method;      use_risk=true
      fi

      MAX_ATTEMPTS=3
      for attempt in $(seq 1 $MAX_ATTEMPTS); do
        [[ $attempt -gt 1 ]] && echo "  >> retry attempt $attempt/$MAX_ATTEMPTS"
        before=$(rows)

        # Launch headless; drop the RTPS_TRANSPORT_SHM spam so logs stay small.
        # Terrain (and crowd) are fixed by the per-seed regen above and identical
        # across the methods; mppi_seed only varies the planner's sampling.
        # The robot spawns at this leg's start pose (outside, or inside a tent).
        ( ros2 launch field_hospital_simulation field_hospital_gz.launch.py \
            headless:=$HEADLESS rviz:=$RVIZ \
            local_mu_blocked:=$LOCAL_MU_BLOCKED \
            fusion_method:=$fmethod \
            use_risk_map:=$use_risk \
            mppi_seed:=$seed \
            spawn_x:=$sx spawn_y:=$sy spawn_yaw:=$syaw \
            metrics:=true metrics_goal:=$goal metrics_out:=$OUT \
            2>&1 | grep --line-buffered -v 'RTPS_TRANSPORT_SHM' \
            > "$OUT/launch_d${density}_${method}_${goal}_s${seed}_a${attempt}.log" ) &

        if ! wait_for_odom; then
          echo "  !! stack did not come up in ${READY_TIMEOUT}s"
          teardown
          [[ $attempt -lt $MAX_ATTEMPTS ]] && continue || echo "  !! giving up after $MAX_ATTEMPTS attempts"
          break
        fi
        sleep "$SETTLE"

        ros2 run field_hospital_simulation send_goal "$goal" >/dev/null 2>&1

        # Wait for metrics_node to append one new row (run finished).
        t=0
        while [[ "$(rows)" -le "$before" && $t -lt $RUN_TIMEOUT ]]; do
          sleep 3; t=$((t+3))
        done

        teardown

        if [[ "$(rows)" -gt "$before" ]]; then
          print_last_metrics
          break    # row written — done with this run
        else
          echo "  !! no row written (timeout, attempt $attempt/$MAX_ATTEMPTS)"
        fi
      done
    done
  done
done
done

echo "Done. Aggregate with:"
echo "  python3 $HERE/aggregate_metrics.py $CSV --baseline independent"
