# Layered Risk Mapping for Autonomous Patient Transport in Expeditionary Medical Facilities

**Project website:** https://lrm-patient-transport.github.io/

## Dependencies

- **OS / ROS 2:** Ubuntu 22.04 + ROS 2 Humble (Python 3.10).
- **Gazebo Sim ("Fortress")**:

  ```bash
  sudo apt install -y ros-humble-ros-gz
  ```

## Installation

```bash
# 1. Create/enter a colcon workspace and clone this repo into src/
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/lagenuina/lrm-patient-transport.git

# 2. Install ROS dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# 3. Install Python dependencies
pip install numpy scipy numba pyyaml

# 4. Build
colcon build --symlink-install

# 5. Source the workspace
source install/setup.bash
```

## Running the simulation

```bash
ros2 launch field_hospital_simulation field_hospital_gz.launch.py
```

This starts Gazebo Sim with the field-hospital world, spawns the wheelchair, and
brings up the full risk stack (elevation, ground-truth terrain/obstacle/crowd reveal, fusion, global + local planners) plus RViz. Useful flags:

```bash
ros2 launch field_hospital_simulation field_hospital_gz.launch.py \
    headless:=true rviz:=false          # no GUI (faster, for batch runs)
```

Then send the robot a goal (see below), or set RViz's Fixed Frame to `map` and use the "2D Goal Pose" tool.

### Sending goals

```bash
ros2 run field_hospital_simulation send_goal <name>       # a named scene goal
ros2 run field_hospital_simulation send_goal 22.5 0.0      # raw x y [yaw]
ros2 run field_hospital_simulation send_goal               # no args: prints the live goal list
```

Named goals come from `maps/scene_meta.json` (20 total: 4 tents × intake / intra-facility / egress):
- `intake_to_{nw,ne,se,sw}` — entrance → ward tent
- `intra_<A>_to_<B>` — ward tent → ward tent (any ordered pair)
- `egress_from_{nw,ne,se,sw}` — ward tent → entrance

### Choose where to spawn

```bash
ros2 launch field_hospital_simulation field_hospital_gz.launch.py leg:=intra_nw_to_ne
```

`leg:=<goal name>` spawns the robot at that goal's start pose (looked up from
`scene_meta.json`) instead of the default outside-the-compound spawn. Or set
`spawn_x`/`spawn_y`/`spawn_yaw` directly.

### Regenerate the scene

Edit the `SCENE DEFINITION` block in `field_hospital_scene.py` (tent layout, terrain patch mix, goals, crowd size, ...), then:

```bash
python3 ~/ros2_ws/src/lrm-patient-transport/scripts/generate_scene.py \
    --seed 7 --risk-density 1.0
colcon build --packages-select field_hospital_simulation
```

- `--seed` — RNG seed for the terrain-patch/slope layout (deterministic; same seed
  ⇒ identical terrain, so different fusion methods can be compared on identical ground)
- `--risk-density` — scales how much risk terrain is packed into the scene (higher ⇒
  more terrain/elevation overlap)
- `--crowd-seed` — separate seed for the wandering crowd (omit for a fresh random
  crowd every generation)
- `--tent-model` — swap the ward-tent mesh; fetch additional Gazebo Fuel models with
  `scripts/fetch_fuel_model.py <owner> "<model name>"`

### Log metrics from a run

```bash
ros2 launch field_hospital_simulation field_hospital_gz.launch.py \
    headless:=true rviz:=false fusion_method:=logodds \
    metrics:=true metrics_goal:=intra_nw_to_ne metrics_out:=~/results
```
```bash
ros2 run field_hospital_simulation send_goal intra_nw_to_ne
```

`metrics_node` appends one row (success, path length, risk exposure, time-to-goal,
...) to `~/sar_metrics/runs.csv` when the run completes, plus a per-run `.npz`
trajectory dump.

### Run a full Monte-Carlo fusion-method comparison

```bash
~/ros2_ws/src/lrm-patient-transport/scripts/run_montecarlo.sh [N_SEEDS [START_SEED]]   # default: 20 seeds from 0
```

For every (seed, goal) pair, runs `independent` / `max` / `logodds` fusion plus the
`norisk` ablation under **identical** conditions (same terrain, same crowd phase, same
MPPI noise draws) so the fusion rule is the only thing that differs. Relaunches the
whole stack per run and tears down fully between runs. Useful env vars:

```bash
RISK_DENSITIES="1.0" ./run_montecarlo.sh      # override the density sweep (default: 0.5 1.0 1.5)
SAR_METRICS_OUT=~/sar_metrics ./run_montecarlo.sh
VIZ=1 ./run_montecarlo.sh                     # show Gazebo GUI + RViz for every run
```