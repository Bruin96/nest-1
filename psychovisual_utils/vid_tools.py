"""Provides tools for processing videos."""

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


def read_video_file(filename, frame_fn=lambda x: x, framen=0):
    """Read the video, returns as a
    numpy array of size (framen, height, width)"""
    from decord import VideoReader, VideoLoader
    from decord import cpu, gpu

    # avoiding the use of GPU here because it may interact
    # with the FPS stability during the experiment
    vr = VideoReader(filename, ctx=cpu(0))
    print(f"# of frames in {filename}: {len(vr)}")
    if framen > 0:
        for i in range(len(vr) // framen - 1):
            yield vr.get_batch(range(i * framen, (i + 1) * framen)).asnumpy()
    else:
        return vr.get_batch(range(len(vr))).asnumpy()


def apply_frame_func(video, frame_fn=lambda x: x):
    result = None
    for frame in video:
        if result is None:
            result = frame_fn(frame)
            result = np.reshape(result, (1,) + result.shape)
        else:
            newframe = frame_fn(frame)
            newframe = np.reshape(newframe, (1,) + newframe.shape)
            result = np.concatenate((result, newframe), axis=0)
    return result


def save_video_file(video, filename, FPS=120):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if filename.lower().endswith(".mp4"):
        ext = ""
    else:
        ext = ".mp4"
    out = cv2.VideoWriter(
        filename + ext, fourcc, FPS, (video.shape[2], video.shape[1])
    )
    for frame in video:
        out.write(frame[:, :, ::-1])
    out.release()


def npy2mp4(directory):
    import os

    for root, _, files in os.walk(directory):
        if root != directory:
            continue
        for file in files:
            clip = np.load(os.path.join(root, file))
            filename, _ = os.path.splitext(file)
            save_video_file(
                clip, os.path.join(root, filename + ".mp4"), FPS=120
            )


if __name__ == "__main__":
    pass
    # npy2mp4("vid_segments/10")
