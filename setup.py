from distutils.version import LooseVersion
import os
from setuptools import find_packages
from setuptools import setup
 
setup(
		name='SVS',
		packages=find_packages(include=["SVS"]),
		version='1.0',
		description='Singing Voice Synthesis',
		author='Jiatong Shi'
		)