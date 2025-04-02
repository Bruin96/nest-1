from psychopy.tools.arraytools import makeRadialMatrix
from psychopy import monitors
import numpy as np


def raisedCos_mask(fringeWidth=0.2, res=128):
    # this code is from psychopy.visual.basevisual.TextureMixin._createTexture()
    hammingLen = 1000  # affects the 'granularity' of the raised cos

    rad = makeRadialMatrix(res)
    intensity = np.zeros_like(rad)
    intensity[np.where(rad < 1)] = 1
    frng = fringeWidth
    raisedCosIdx = np.where([np.logical_and(rad <= 1, rad >= 1 - frng)])[1:]

    # Make a raised_cos (half a hamming window):
    raisedCos = np.hamming(hammingLen)[: hammingLen // 2]
    raisedCos -= np.min(raisedCos)
    raisedCos /= np.max(raisedCos)

    # Measure the distance from the edge - this is your index into the
    # hamming window:
    dFromEdge = np.abs((1 - fringeWidth) - rad[raisedCosIdx])
    dFromEdge /= np.max(dFromEdge)
    dFromEdge *= np.round(hammingLen / 2)

    # This is the indices into the hamming (larger for small distances
    # from the edge!):
    portionIdx = (-1 * dFromEdge).astype(int)

    # Apply the raised cos to this portion:
    intensity[raisedCosIdx] = raisedCos[portionIdx]

    # Scale it into the interval -1:1:
    intensity = intensity - 0.5
    intensity /= np.max(intensity)

    # Sometimes there are some remaining artifacts from this process,
    # get rid of them:
    artifactIdx = np.where(np.logical_and(intensity == -1, rad < 0.99))
    intensity[artifactIdx] = 1
    artifactIdx = np.where(np.logical_and(intensity == 1, rad > 0.99))
    intensity[artifactIdx] = 0
    return (intensity + 1) / 2


def linearize(patch, gammaGrid):
    # input must be between 0 and 1
    # use luminance conversion
    _, _, gamma, a, b, k = gammaGrid[0]
    patch_clip = np.clip(patch, 0, 1)
    f = lambda x: a + (b + k * x) ** gamma
    return f(patch_clip)


def inv_linearize(patch, gammaGrid):
    # use inverse luminance conversion
    minl, maxl, gamma, a, b, k = gammaGrid[0]
    patch_clip = np.clip(patch, minl, maxl)
    f = lambda y: ((y - a) ** (1 / gamma) - b) / k
    return f(patch_clip)


def apply_mask(patch, gammaGrid=None, bg_level=0.0):
    if gammaGrid is None:
        mon = monitors.Monitor(
            "testMonitor"
        )  # fetch the most recent calib for this monitor
        calibs = mon.calibs
        mon.setCurrent(list(calibs.keys())[-1])  # the most recent calibration
        gammaGrid = calibs[list(calibs.keys())[-1]]["gammaGrid"]
    scale = lambda x: (x + 1) / 2
    inv_scale = lambda y: (y * 2) - 1
    patch_scaled = scale(patch)
    patch_lin = linearize(patch_scaled, gammaGrid)
    bg_scaled = scale(bg_level)
    bglum = linearize(bg_scaled, gammaGrid)
    mask = raisedCos_mask(fringeWidth=0.2, res=patch_lin.shape[0])
    patch_masked = mask * patch_lin + (1 - mask) * bglum
    return inv_scale(inv_linearize(patch_masked, gammaGrid))


if __name__ == "__main__":
    mask = raisedCos_mask(fringeWidth=0.2, res=71)
    print(mask)
