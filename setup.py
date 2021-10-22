# Copyright 2021 The DJIPupperSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for DJIPupperSim.

Install for development:

  pip intall -e .
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name="puppersim",
    version="0.0.1",
    description=("A simulation of the DJI Pupper robot"),
    author="Pupper Sim Authors",
    author_email="no-reply@google.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="http://github.com/google/brax",
    license="Apache 2.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={'data': ["upper_leg_left_rear.obj"]},
    install_requires=[
        "pybullet",
        "attrs",
        "gym",
        "numpy",
        "absl-py",
        "gin-config",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="Pupper Simulation quadruped robot reinforcement learning rigidbody physics",
)
