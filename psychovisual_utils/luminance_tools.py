"""Provides tools for computing luminance and gamma correction."""

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


def rgb2lum(frame):
    """Convert sRGB to relative luminance (Y).
    Expects input with size (h, w, 3)"""
    # if f.max() > 1:
    #     frame = f.astype(np.float32) / 255.0
    # else:
    #     frame = f
    gamma_low = lambda u: u / 12.92
    gamma_high = lambda u: np.power((u + 0.055) / 1.055, 2.4)
    gamma_transform = lambda u: np.where(
        u <= 0.04045, gamma_low(u), gamma_high(u)
    )
    return (
        0.21263901 * gamma_transform(frame[:, :, 0])
        + 0.71516868 * gamma_transform(frame[:, :, 1])
        + 0.07219232 * gamma_transform(frame[:, :, 2])
    )


def gamma22(frame):
    frame_weights = np.array([0.21263901, 0.71516868, 0.07219232])
    frame_weights = np.reshape(frame_weights, (1, 1, 3))
    return np.sum(np.power(frame, 2.2) * frame_weights, axis=2)


def apply_gamma(inp):
    """Apply display gamma transformation.
    Expects (h, w, 3) as input. The transformation is
    display-specific (LG OLED AI ThinQ). Has to be
    computed for display brightness, contrast, color
    settings.
    This function has to be updated after changing
    display settings.
    """
    # input range must be in [0-1]
    # if inp.max() > 1:
    #     rgb = inp.astype(np.float16) / 255.0
    # else:
    #     rgb = inp
    # full linearization: a + (b + k*x) ** g
    a_ = np.array([0.2619, 0.2823, 0.3229, 0.2201])
    b_ = np.array([0.3200, 0.2509, 0.1533, 0.3659])
    k_ = np.array([4.4953, 9.2361, 2.7293, 10.8083])
    g_ = np.array([2.2545, 2.0813, 2.2183, 2.1222])
    peak_lum = 154
    lin = (
        lambda a, b, k, g, mat, idx: a[idx]
        + (b[idx] + k[idx] * mat[:, :, idx]) ** g[idx]
    )
    add_dim = lambda x: np.reshape(x, x.shape + (1,))
    lums = [add_dim(lin(a_, b_, k_, g_, inp, int(e))) for e in list(range(3))]
    result = np.concatenate(lums, axis=2)
    # result = None
    # for ch in lums:
    #     if result is None:
    #         result = ch
    #     else:
    #         result = np.concatenate((result, ch), axis=2)
    return np.sum(result, axis=2) / peak_lum


def test_applygamma():
    levels = np.arange(0, 1, 0.01)
    for level in levels:
        frame = np.ones((1, 1, 3)) * level
        print(
            f"level: {level} gamma corrected: {apply_gamma(frame)} rgb2lum: {rgb2lum(frame)} gamma22: {gamma22(frame)}"
        )


if __name__ == "__main__":
    test_applygamma()
