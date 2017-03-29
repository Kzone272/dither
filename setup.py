# import sys
# from cx_Freeze import setup, Executable
from distutils.core import setup
import py2exe

# addtional_mods = ['numpy.core._methods', 'numpy.lib.format', 'scipy']
addtional_mods = []

setup(
    name = "Dither",
    version = "1.0",
    description = "A tool for dithering images in various ways.",
    console=['dither.py'])
    # options = {'build_exe': {'includes': addtional_mods}},
    # executables = [Executable("Dither.py", base = "Win32GUI")])
