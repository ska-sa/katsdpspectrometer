#!/usr/bin/env python3

###############################################################################
# SKA South Africa (http://ska.ac.za/)                                        #
# Author: spt@ska.ac.za                                                       #
# Copyright (c) 2018, National Research Foundation (Square Kilometre Array).  #
# All rights reserved.                                                        #
#                                                                             #
# THIS SOFTWARE MAY NOT BE COPIED OR DISTRIBUTED IN ANY FORM WITHOUT THE      #
# WRITTEN PERMISSION OF SKA SA.                                               #
###############################################################################

from setuptools import setup, find_packages


setup(
    name='katsdpspectrometer',
    description='Karoo Array Telescope Digitiser Spectrometer',
    author='Ludwig Schwardt',
    author_email='ludwig@ska.ac.za',
    url='https://github.com/ska-sa/katsdpingest',
    packages=find_packages(),
    scripts=[
        'scripts/spectrometer.py'
        ],
    python_requires='>=3.5',
    setup_requires=['katversion'],
    install_requires=[
        'aiokatcp',
        'katsdptelstate',
        'katsdpservices',
        'numpy',
        'spead2>=1.5.0',     # For stop_on_stop_item
    ],
    use_katversion=True
)
