"""Provides tools related to physical display parameters."""

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

import numpy as np


supported_displays = {
    "lg": {  # LG OLED AI ThinQ
            "Name": "lg",
            "ViewingDist": 0.65,
            "Width": 1.20,
            "Height": 0.675,
            "PeakLum": 160,
            "PxHoriz": 3840,
            "PxVert": 2160,
            "RefreshRate": 120
        },
    "macbookpro16_2019": {  # Macbook Pro 16 2019
            "Name": "macbookpro16_2019",
            "ViewingDist": 0.40,
            "Width": 0.345,
            "Height": 0.215,
            "PeakLum": 250,
            "PxHoriz": 3072,
            "PxVert": 1920,
            "RefreshRate": 60
    },
    "macbookpro13_2021": {  # Macbook Pro 13 2021
            "Name": "macbookpro13_2021",
            "ViewingDist": 0.40,
            "Width": 0.286,
            "Height": 0.179,
            "PeakLum": 250,
            "PxHoriz": 1440,
            "PxVert": 900,
            "RefreshRate": 60
    },
    "dell_u2718q": {  # Dell U2718Q 27" 4K Display
            "Name": "dell_u2718q",
            "ViewingDist": 0.60,
            "Width": 0.596736,
            "Height": 0.335664,
            "PeakLum": 250,
            "PxHoriz": 3840,
            "PxVert": 2160,
            "RefreshRate": 60
    },
    "acer_x27": {  # Acer X27 27" 4K Display
            "Name": "acer_x27",
            "ViewingDist": 0.60,
            "Width": 0.5952,
            "Height": 0.3348,
            "PeakLum": 600,
            "PxHoriz": 3840,
            "PxVert": 2160,
            "RefreshRate": 144
    }
}


def get_display_params(display):
    display_str = display.lower()
    if display_str in supported_displays.keys():
        params = supported_displays[display_str]
    else:
        raise ValueError
    px_size = params["Width"] / params["PxHoriz"]
    view_dist_px = params["ViewingDist"] / px_size
    peak_cpd = np.tan(np.deg2rad(0.5)) * view_dist_px
    params["PeakCPD"] = peak_cpd
    params["ViewingDist_px"] = view_dist_px
    return params
    
    
def generate_display_params(display, peak_lum, refresh_rate):
    params = {}
    params["Name"] = display.name
    params["ViewingDist"] = display.currentCalib["distance"] / 100.0 # in meters
    params["Width"] = display.currentCalib["width"] / 100.0 # in meters
    params["Height"] = 9.0 / 16.0 * params["Width"]
    params["PeakLum"] = peak_lum if peak_lum > 0 else 2.0 * display.currentCalib["meanLum"]
    
    px_horiz, px_vert = display.getSizePix()
    params["PxHoriz"] = px_horiz
    params["PxVert"] = px_vert
    
    params["RefreshRate"] = refresh_rate
    
    px_size = params["Width"] / params["PxHoriz"]
    view_dist_px = params["ViewingDist"] / px_size
    peak_cpd = np.tan(np.deg2rad(0.5)) * view_dist_px
    params["PeakCPD"] = peak_cpd
    params["ViewingDist_px"] = view_dist_px
    
    print(f"params: {params}")
    
    return params


def eccentricity_map(params, frame_size=None, grid_size=None):
    """ frame_size = (frame_height, frame_width) """
    px_size = params["Width"] / params["PxHoriz"]
    view_dist_px = params["ViewingDist"] / px_size
    if frame_size is None:
        x = np.linspace(0, params["PxHoriz"] - 1, params["PxHoriz"])
        y = np.linspace(0, params["PxVert"] - 1, params["PxVert"])
        x = np.abs(x - params["PxHoriz"] / 2)
        y = np.abs(y - params["PxVert"] / 2)
    else:
        x = np.linspace(0, frame_size[1] - 1, frame_size[1])
        y = np.linspace(0, frame_size[0] - 1, frame_size[0])
        x = np.abs(x - frame_size[1] / 2)
        y = np.abs(y - frame_size[0] / 2)
    xv, yv = np.meshgrid(x, y)
    euc_px = np.sqrt(np.square(xv) + np.square(yv))
    map_radian = np.arctan(euc_px / view_dist_px)
    map_deg = np.rad2deg(map_radian)
    if grid_size is not None:
        thickness = 3
        current_row = 0
        while current_row < map_deg.shape[0]:
            for t in range(thickness):
                if current_row + t >= map_deg.shape[0]:
                    break
                map_deg[current_row + t, :] = 0
            current_row += grid_size[0]
        current_col = 0
        while current_col < map_deg.shape[1]:
            for t in range(thickness):
                if current_col + t >= map_deg.shape[1]:
                    break
                map_deg[:, current_col + t] = 0
            current_col += grid_size[1]
    return map_deg


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = eccentricity_map(
        get_display_params("dell_u2718q"), grid_size=[142, 142]
    )
    plt.matshow(a)
    plt.show()
