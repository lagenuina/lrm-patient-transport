# Third-party mesh models

These prop meshes are cached from the [Gazebo Fuel](https://app.gazebosim.org/fuel)
model library, authored by **Open Robotics**, for the photorealistic Gazebo Sim
(Fortress) field-hospital world. They are redistributed here under their
original licenses (Creative Commons Attribution — CC-BY 4.0). See each model's
`model.config` for the precise license/author metadata.

| Local dir          | Fuel model                       | Use in scene        |
|--------------------|----------------------------------|---------------------|
| `tent/`            | OpenRobotics/tent                | ward tents          |
| `TrolleyBed/`      | OpenRobotics/TrolleyBed          | cots / patient beds |
| `person_standing/` | OpenRobotics/Standing person     | staff / patients    |
| `IVStand/`         | OpenRobotics/IVStand             | equipment           |

Re-download (e.g. to refresh or add models):

    curl -sL "https://fuel.gazebosim.org/1.0/OpenRobotics/models/tent/tip/tent.zip" -o tent.zip

Cite the asset authors if you publish renders/figures using them.
