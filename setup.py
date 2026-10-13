import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'field_hospital_simulation'


def _recursive_data(src_root, dest_root):
    """Mirror a directory tree (e.g. models/) into share/, preserving subdirs,
    so colcon installs every mesh/texture/config file at the right relative path
    for model:// resolution."""
    entries = []
    for dirpath, _dirs, files in os.walk(src_root):
        if not files:
            continue
        rel = os.path.relpath(dirpath, src_root)
        dest = os.path.join(dest_root, rel) if rel != '.' else dest_root
        entries.append((dest, [os.path.join(dirpath, f) for f in files]))
    return entries

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'),   glob('urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'maps'),   glob('maps/*')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'media', 'materials', 'textures'),
         glob('media/materials/textures/*')),
        (os.path.join('lib', package_name), glob('scripts/*.py')),
    ] + _recursive_data('models', os.path.join('share', package_name, 'models')),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorena Genua',
    maintainer_email='genua.l@northeastern.edu',
    description='Outbreak field-hospital Gazebo sim for the risk_maps stack.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_terrain  = field_hospital_simulation.sim_terrain_node:main',
            'sim_obstacle = field_hospital_simulation.sim_obstacle_node:main',
            'sim_crowd    = field_hospital_simulation.sim_crowd_node:main',
            'send_goal    = field_hospital_simulation.send_goal:main',
            'odom_tf      = field_hospital_simulation.odom_tf_node:main',
            'metrics      = field_hospital_simulation.metrics_node:main',
        ],
    },
)
