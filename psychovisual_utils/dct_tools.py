"""Provides tools for DCT computation."""

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

import threading

import numpy as np
import png
from psychopy import data
from scipy.fftpack import dctn, idctn


def dctn1(x):
    result = dctn(x, type=1) / (
        np.prod([e - 1 for e in x.shape]) * (2 ** x.ndim)
    )
    return result


def idctn1(x):
    return idctn(x, type=1)


def dct2contrast(x):
    amplitude = np.copy(x) * (2 ** x.ndim)
    for z_ in [0, x.shape[0] - 1]:
        amplitude[z_, :, :] /= 2
    for x_ in [0, x.shape[1] - 1]:
        amplitude[:, x_, :] /= 2
    for y_ in [0, x.shape[2] - 1]:
        amplitude[:, :, y_] /= 2
    # return np.abs(contrast)
    # return np.abs(contrast) / 0.5
    return np.abs(amplitude) / np.maximum(np.abs(amplitude[0, 0, 0]), 0.3)
    # return np.abs(contrast) / (np.log(1 + np.exp(contrast[0, 0, 0])) - 0.3132616875182228)


# def dct3(mat):
#    # assuming mat[z,x,y] order of indices
#    result = np.zeros(mat.shape)
#    C_u = np.ones(mat.shape)
#    C_v = np.ones(mat.shape)
#    C_w = np.ones(mat.shape)
#    C_u[0, 0, 0] = 0.5 ** 0.5
#    C_v[0, 0, 0] = 0.5 ** 0.5
#    C_w[0, 0, 0] = 0.5 ** 0.5
#    C_u, C_v, C_w = C_u / 2, C_v / 2, C_w / 2
#    cos_prod = lambda x, y, z, u, v, w: \
#        np.cos((2 * x + 1) * u * np.pi) / (2 * mat.shape[1]) * \
#        np.cos((2 * y + 1) * v * np.pi) / (2 * mat.shape[2]) * \
#        np.cos((2 * z + 1) * w * np.pi) / (2 * mat.shape[0])
#    process_length = np.prod(mat.shape) ** 2
#    n = 1
#    for w_ in range(mat.shape[0]):
#        for u_ in range(mat.shape[1]):
#            for v_ in range(mat.shape[2]):
#                for z_ in range(mat.shape[0]):
#                    for x_ in range(mat.shape[1]):
#                        for y_ in range(mat.shape[2]):
#                            prod = cos_prod(x_, y_, z_, u_, v_, w_)
#                            result[w_, u_, v_] += \
#                                C_u[w_, u_, v_] * \
#                                C_v[w_, u_, v_] * \
#                                C_w[w_, u_, v_] * \
#                                prod * \
#                                mat[z_, x_, y_]
#                            print(f"Processed {n/process_length * 100}%\b")
#                            n += 1
#    return result


class dctThread(threading.Thread):
    def __init__(self, im, inverse=False):
        threading.Thread.__init__(self)
        self.im = im
        self.inverse = inverse
        self.result = None

    def run(self):
        if self.inverse:
            self.result = idctn1(self.im)
        else:
            self.result = dctn1(self.im)

    def get_result(self):
        if self.result is None:
            self.join()
        return self.result


def create_dct_thread(im, inverse=False):
    t = dctThread(im, inverse=inverse)
    t.start()
    return t


def generate_stimulus(sz, x, y, z, val):
    vid_dct = dctn1(np.zeros(sz))
    # averaged signal should have half of the max display luminance
    vid_dct[0, 0, 0] = 0.5
    vid_dct[z, x, y] = val

    # scale the threshold for DCT-I
    if z not in [0, vid_dct.shape[0] - 1]:
        vid_dct[z, x, y] /= 2.0
    if x not in [0, vid_dct.shape[1] - 1]:
        vid_dct[z, x, y] /= 2.0
    if y not in [0, vid_dct.shape[2] - 1]:
        vid_dct[z, x, y] /= 2.0
    # if z % 2 == 1:
    #     print((f"Warning: using an odd int as the temporal index ({z})"
    #            ", the signal will be aperiodic."))
    vid_idct = idctn1(vid_dct)
    return vid_idct


def save_stimuli(sz, zxy_tuple, val):
    from external import raisedCos_mask

    mask = raisedCos_mask(res=71)
    for zxy in zxy_tuple:
        z, x, y = zxy
        stim = generate_stimulus(sz, x, y, z, val)
        for idx_z in range(stim.shape[0]):
            stim[idx_z] = (stim[idx_z] - 0.5) * mask + 0.5
        for idx_z in range(sz[0]):
            filename = (
                f"z{sz[0]}_x{sz[1]}_y{sz[2]}_"
                f"_{z}_{x}_{y}_frame{idx_z:03d}.png"
            )
            png.from_array((stim[idx_z] * 255).astype(np.uint8), "L").save(
                filename
            )


def visualize_stimuli(sz, zxy_tuple, val, win, rows, cols, save_n_frames=0):
    from psychopy import event, visual

    patches = []
    window_size = win.size
    row_dist = window_size[1] / (rows + 2)
    col_dist = window_size[0] / (cols + 2)
    for (i, zxy) in enumerate(zxy_tuple):
        row_idx = i // cols
        col_idx = i % cols
        z, x, y = zxy
        
        vid_sinus = np.ones((60, sz[1], sz[2]))
        for i in range(sz[1]):
            for j in range(sz[2]):
                vid_sinus[:, i, j] = 0.5 + val*np.sin(2*np.pi*(x/2.0)*float(i)/(sz[1]-1))
        
        stim = vid_sinus
        #stim = generate_stimulus(sz, x, y, z, val)
        patches.append([])
        for z_ in range(sz[0]):
            frame = stim[z_, 0: sz[1] - 1, 0: sz[2] - 1]
            frame = (np.round(frame * 254) / 255 - 0.5) * 2
            patches[-1].append(
                visual.ImageStim(
                    win,
                    image=frame,
                    mask="raisedCos",#"raisedCos",
                    units="pix",
                    pos=(
                        round(
                            col_dist * col_idx - col_dist * ((cols - 1) / 2)
                        ),
                        round(
                            row_dist * ((rows - 1) / 2) - row_dist * row_idx
                        ),
                    ),
                    size=(sz[1] - 1, sz[2] - 1),
                    ori=0.0,
                    color=(1.0, 1.0, 1.0),
                    colorSpace="rgb",
                    contrast=1.0,
                    opacity=1.0,
                    depth=0,
                    interpolate=False,
                    flipHoriz=False,
                    flipVert=False,
                    name=None,
                    autoLog=False,
                    # {'fringeWidth': 0.001, 'sd': 3} for "raisedCos"
                    maskParams=None,#{'fringeWidth': 0.001, 'sd': 3},
                )
            )
    win.flip()
    frame_idx = 0
    direction = 1
    keep_visualizing = True
    pause = False
    next_frame = False
    save_frame_idx = 0
    while keep_visualizing:
        if (not pause) or next_frame:
            frame_idx += direction
            if frame_idx >= (sz[0] - 1):
                direction = -1
            elif frame_idx == 0:
                direction = 1
            for patch in patches:
                patch[frame_idx].draw()
            win.flip()
            if save_n_frames > 0:
                buffer = win.getMovieFrame(buffer="front")
                buffer.save(
                        f"screenshot_frame_{save_frame_idx:06d}.png",
                        "PNG",
                    )
                save_frame_idx += 1
                save_n_frames -= 1
            if next_frame:
                next_frame = False
        keylist = event.getKeys()
        if keylist:
            for key in keylist:
                if key == "q":
                    keep_visualizing = False
                elif key == "p":
                    pause = not pause
                elif key == "n":
                    next_frame = True
                elif key == "c":
                    # save the current frame to disk
                    buffer = win.getMovieFrame(buffer="front")
                    buffer.save(
                        f"screenshot_{data.getDateStr(format='%Y-%m-%d-%H%M%S')}.png",
                        "PNG",
                    )
            event.clearEvents()
    win.flip()


def main():
    sz = (25, 71, 71)
    threshold = 0.50
    save_stimuli(sz, ((12, 17, 0), (12, 0, 17), (12, 17, 17)), threshold)


def test_dctn():
    tmp = np.random.random((25, 71, 71))
    tmpdct = dctn1(tmp)
    tmpdcti = idctn1(tmpdct)
    diff = abs(tmp - tmpdcti)
    print(diff)
    print(diff.max())


def test_dct2contrast():
    sz = (25, 71, 71)
    (x, y, z) = (0, 0, 12)
    val = 0.5
    stimcontrast_orig = np.zeros(sz)
    stimcontrast_orig[0, 0, 0] = 0.5
    stimcontrast_orig[z, x, y] = val
    stim = idctn1(stimcontrast_orig)
    # stim = generate_stimulus(sz, x, y, z, val)
    stimdct = dctn1(stim)
    stimcontrast = dct2contrast(stimdct)
    absdiff = np.abs(stimcontrast - stimcontrast_orig)
    print(f"{absdiff.max()}")


if __name__ == "__main__":
    sz = (25, 71, 71)
    val = 0.5
    z = (4, 6, 12, 24)
    x = (0, round((sz[1] - 1) / 4), round((sz[1] - 1) / 2))
    y = (0, round((sz[2] - 1) / 4), round((sz[2] - 1) / 2))
    zxy_tuple = []
    for z_ in z:
        for x_ in x:
            for y_ in y:
                zxy_tuple.append((z_, x_, y_))
    save_stimuli(sz, zxy_tuple, val)
    # test_dct2contrast()
