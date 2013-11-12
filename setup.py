#!/usr/bin/env python

from setuptools import setup

setup(name='Reverse Game of Life',
    version='0.0',
    description='Python Distribution Utilities',
    url='https://github.com/valisc/reverse-game-of-life',
    packages=['reverse_game_of_life'],
    install_requires=open("dependencies.txt").readlines() 
    )

