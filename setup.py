#! /usr/bin/env python

from distutils.core import setup, Extension

module1 = Extension("fastgroup", sources=['fastgroup.cpp']);

setup(name='fastgroup',
        version='1.0',
        description='fast like-pixel grouping algorithm',
        ext_modules=[module1]);
