#!/usr/bin/env python3
"""sdf_world_gz.py — build a Gazebo Sim (Fortress / ogre2 PBR) world from a
field_hospital_scene.Scene, using real downloaded mesh models for the props and
a tiled, textured ground so it reads as an actual field-hospital site rather
than coloured blocks.

Mesh models (cached under the package models/ dir, referenced as model://):
    tent             -> ward tents (2 placed per ward footprint, unscaled)
    TrolleyBed       -> cots / patient beds
    person_standing  -> staff / patients
    IVStand          -> equipment carts

The navigation pipeline is untouched: the GT obstacle grid still comes from the
scene footprints (field_hospital_scene.rasterise), so what the planner avoids is
unchanged — only the rendering differs. Each mesh model carries its own
collision; perimeter walls stay as PBR boxes. Zone floor tints (green/yellow/red
+ decon) are kept as translucent slabs over the grass for the semantic story.

gz resolves relative texture URIs *relative to the world file*, so they are
written as ../media/... (worlds/../media -> share/field_hospital_simulation/media). model://
prop meshes resolve via the models/ dir the launch adds to the resource path.
"""

from __future__ import annotations

TEX = '../media/materials/textures'


def _mat_tex(name, rough=0.95, metal=0.0, normal=True):
    nm = f'<normal_map>{TEX}/{name}_normal.png</normal_map>' if normal else ''
    return f'''<material>
        <diffuse>1 1 1 1</diffuse><specular>0.15 0.15 0.15 1</specular>
        <pbr><metal>
          <albedo_map>{TEX}/{name}_albedo.png</albedo_map>{nm}
          <roughness>{rough}</roughness><metalness>{metal}</metalness>
        </metal></pbr>
      </material>'''


def _mat_color(r, g, b, a=1.0, rough=0.8, metal=0.0):
    return f'''<material>
        <ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse>
        <specular>0.1 0.1 0.1 1</specular>
        <pbr><metal><roughness>{rough}</roughness><metalness>{metal}</metalness></metal></pbr>
      </material>'''


def _box(sx, sy, sz):
    return f'<box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box>'


def _include(uri, name, x, y, z=0.0, yaw=0.0, static=True):
    st = '<static>true</static>' if static else ''
    return f'''
    <include>
      <uri>{uri}</uri><name>{name}</name>{st}
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>
    </include>'''


# --------------------------------------------------------------------------
# props -> mesh includes (one or more per scene Box)
# --------------------------------------------------------------------------
def _box_models(b, tent_model='tent'):
    if b.name.startswith('tentwall'):
        # Invisible collision-only wall (the tent mesh provides the visual).
        return f'''
    <model name="{b.name}"><static>true</static>
      <pose>{b.cx:.3f} {b.cy:.3f} 0 0 0 0</pose>
      <link name="link"><collision name="c">
        <pose>0 0 {b.sz/2:.3f} 0 0 0</pose>
        <geometry>{_box(b.sx, b.sy, b.sz)}</geometry>
      </collision></link></model>'''
    if b.name.startswith('tent'):
        # Visual-only marquee (collision comes from the tentwall_* boxes).
        # Calibrated to the current military_tent model.sdf, whose internal pose
        # puts the doorway on the -x short end at include-yaw 0; add 180 deg to
        # move it to the +x end. Driven by the SAME tent_door_sign knob as the GT
        # door, so the mesh opening and the functional door always line up.
        # NOTE: if model.sdf's internal orientation is flipped, flip this too.
        # name = tent_{row}_{col}.
        from . import field_hospital_scene as fhs
        _, row, col = b.name.split('_')
        yaw = 3.14159 if fhs.tent_door_sign(row, int(col)) > 0 else 0.0
        return _include(f'model://{tent_model}', b.name, b.cx, b.cy, yaw=yaw)
    if b.name.startswith('cot'):
        # Bed length (x) along the footprint's long axis.
        yaw = 0.0 if b.sx >= b.sy else 1.5708
        return _include('model://TrolleyBed', b.name, b.cx, b.cy, yaw=yaw)
    if b.name.startswith('person'):
        return _include('model://person_standing', b.name, b.cx, b.cy)
    if b.name.startswith('cart'):
        return _include('model://IVStand', b.name, b.cx, b.cy)
    # walls (and anything else): PBR box matching the footprint
    mat = _mat_tex('metal', rough=0.6, metal=0.7) if 'wall' in b.name \
        else _mat_color(*b.rgba[:3])
    return f'''
    <model name="{b.name}"><static>true</static>
      <pose>{b.cx:.3f} {b.cy:.3f} 0 0 0 0</pose>
      <link name="link">
        <collision name="c"><pose>0 0 {b.sz/2:.3f} 0 0 0</pose>
          <geometry>{_box(b.sx, b.sy, b.sz)}</geometry></collision>
        <visual name="v"><pose>0 0 {b.sz/2:.3f} 0 0 0</pose>
          <geometry>{_box(b.sx, b.sy, b.sz)}</geometry>{mat}</visual>
      </link>
    </model>'''


WALK_DAE = '/usr/share/gazebo-11/media/models/walk.dae'
# walk.dae is a true-to-life ~1.7 m adult, but the wheelchair mesh was shrunk
# to 85% (mesh_scale in wheelchair_gz.urdf.xacro, for tent-door clearance), so
# unscaled people loomed oversized next to it. Shrink to match — skin only:
# Ignition's own actor.sdf demo for this exact file scales only the skin and
# leaves <animation> at its default 1.0; scaling the animation skeleton too
# compounds with the skin scale through the leg joint chain and sinks the
# feet through the floor (confirmed empirically).
PERSON_SCALE = 0.8
# The walk.dae actor's origin sits ~1 m up its body, so waypoints at z=0 sink
# the legs below the floor (you'd see it chest-up). Lift the whole path so the
# feet land on the ground. This offset comes from the (unscaled) skeleton, not
# the skin mesh, so it does NOT scale with PERSON_SCALE. Visual only — the
# obstacle grid uses x/y. Nudge this if the people float (lower) or sink
# (raise).
WALK_Z = 1.0


def _crowd_model(c):
    """A walking-crowd person as a Gazebo actor: the walk.dae skin animates the
    legs, and a looping waypoint trajectory walks it around its home patch.
    Actors are kinematic by construction (never fall) and loop forever (never
    freeze), and the waypoints are bounded (never fly off). The matching
    obstacle footprint is published by sim_crowd_node, which replays the same
    waypoints into the grid so the planner avoids them."""
    name = c['name']
    wps = c.get('waypoints') or [[0.0, c['x'], c['y'], c.get('yaw', 0.0)]]
    pts = '\n'.join(
        f'        <waypoint><time>{t:.2f}</time>'
        f'<pose>{x:.3f} {y:.3f} {WALK_Z} 0 0 {yaw:.3f}</pose></waypoint>'
        for t, x, y, yaw in wps)
    return f'''
    <actor name="{name}">
      <skin><filename>{WALK_DAE}</filename><scale>{PERSON_SCALE}</scale></skin>
      <animation name="walking">
        <filename>{WALK_DAE}</filename><interpolate_x>true</interpolate_x>
      </animation>
      <script>
        <loop>true</loop>
        <auto_start>true</auto_start>
        <delay_start>0.0</delay_start>
        <trajectory id="0" type="walking">
{pts}
        </trajectory>
      </script>
    </actor>'''


def _zone_tint(z, idx):
    r, g, b, a = z.rgba
    if a <= 0.0:
        return ''
    cx, cy = (z.xmin + z.xmax) / 2.0, (z.ymin + z.ymax) / 2.0
    sx, sy = max(0.01, z.xmax - z.xmin), max(0.01, z.ymax - z.ymin)
    zz = 0.02 + idx * 0.004
    return f'''
    <model name="zone_{idx}_{z.label or 'z'}"><static>true</static>
      <pose>{cx:.3f} {cy:.3f} {zz:.4f} 0 0 0</pose>
      <link name="link"><visual name="v">
        <geometry>{_box(sx, sy, 0.004)}</geometry>
        {_mat_color(r, g, b, a * 0.5, rough=0.95)}
        <transparency>{1.0 - a * 0.5:.2f}</transparency>
      </visual></link></model>'''


def _ground_tiles():
    """Tiled grass ground so the texture repeats at a natural scale (a single
    big plane would stretch one texture across the whole site). Covers the
    map extent [X0,X1]x[Y0,Y1] plus a margin (tracks the scene, so widening the
    world widens the grass); each tile is a thin PBR-grass box."""
    import math
    from .field_hospital_scene import X0, X1, Y0, Y1
    tile = 6.0

    def _axis(lo, hi):
        start = math.floor((lo - 3.0) / tile) * tile     # snap to the tile grid
        n = int(math.ceil((hi + 3.0 - start) / tile))    # extend past hi + margin
        return [start + i * tile for i in range(n)]

    xs = _axis(X0, X1)
    ys = _axis(Y0, Y1)
    out = []
    n = 0
    for x in xs:
        for y in ys:
            out.append(f'''
    <model name="ground_{n}"><static>true</static>
      <pose>{x + tile/2:.2f} {y + tile/2:.2f} 0 0 0 0</pose>
      <link name="link">
        <visual name="v"><geometry>{_box(tile, tile, 0.02)}</geometry>
          {_mat_tex('grass', rough=1.0)}</visual>
      </link></model>''')
            n += 1
    # one big flat collision plane underneath (so physics is a clean floor)
    collide = '''
    <model name="ground_collision"><static>true</static>
      <link name="link"><collision name="c">
        <geometry><plane><normal>0 0 1</normal><size>120 120</size></plane></geometry>
      </collision></link></model>'''
    return ''.join(out) + collide


def build_world_gz(scene, tent_model='tent') -> str:
    zones = ''.join(_zone_tint(z, i) for i, z in enumerate(scene.zones))
    models = ''.join(_box_models(b, tent_model) for b in scene.boxes)
    crowd = ''.join(_crowd_model(c) for c in scene.crowd)
    ground = _ground_tiles()
    return f'''<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="field_hospital">
    <physics name="default" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="ignition-gazebo-physics-system"
            name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system"
            name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system"
            name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system"
            name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.7 0.8 0.95 1</background>
      <sky></sky>
      <grid>false</grid>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>8 4 12 0 0 0</pose>
      <diffuse>1.0 0.98 0.95 1</diffuse>
      <specular>0.35 0.35 0.35 1</specular>
      <direction>-0.4 -0.3 -0.9</direction>
      <attenuation><range>100</range><constant>0.9</constant>
        <linear>0.01</linear><quadratic>0.001</quadratic></attenuation>
    </light>
{ground}
{zones}
{models}
{crowd}
  </world>
</sdf>
'''
