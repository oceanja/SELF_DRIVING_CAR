import os
from setuptools import setup, find_packages

# export PYTHONPATH=$PYTHONPATH:/Users/bchippada/Desktop/FSD_Final_Demo

# Automatically create necessary directories
dirs = [
    'src/models',
    'src/inference',
    'data',
    'saved_models/regression_model',
    'saved_models/lane_segmentation_model',
    'saved_models/object_detection_model',
    'utils',
    'notebooks',
    'tests'
]

for dir in dirs:
    os.makedirs(dir, exist_ok=True)

# Read dependencies
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

# Setup configuration
setup(
    name='self_driving_car_project',
    version='0.3',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=required_packages,
    entry_points={
        'console_scripts': [
            'run_fsd_inference=src.inference.run_fsd_inference:main',
            'run_segmentation=src.inference.run_segmentation_obj_det:main',
            'run_steering=src.inference.run_steering_angle_prediction:main'
        ]
    },
    author='Aditya Meshram',
    description='A self driving car project using computer vision and deep learning',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Recruiters',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ],
    python_requires='>=3.8',
)