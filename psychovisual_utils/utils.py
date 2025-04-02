"""Provides utilities for miscellaneous stuff."""

__version__ = "0.1"
__author__ = "Cara Tursun"
__copyright__ = """Copyright (c) 2022 Cara Tursun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
__license__ = "MIT"

import subprocess
import time


def get_luminance():
    """Measure the luminance using Argyll_V2"""
    pathToArgyll = "D:/Cara/tools/Argyll_V2.1.2/bin"
    cmd = "spotread -e -x -O -y o"
    returned_output = subprocess.check_output(
        cmd, shell=True, cwd=pathToArgyll
    ).decode("utf-8")
    lines = returned_output.split("\n")
    for line in lines:
        if "Result is XYZ: " not in line:
            continue
        else:
            s = line.strip()[15:]
            fields = s.split(", Yxy: ")
            # XYZ = [float(e) for e in fields[0].split(" ")]
            Yxy = [float(e) for e in fields[1].split(" ")]
            return Yxy[0]
    return -1


def reset_display(win):
    bg_color = win.color
    win.color = (-1, -1, -1)
    win.flip()
    time.sleep(0.1)
    win.color = bg_color
    win.flip()
    time.sleep(0.1)
