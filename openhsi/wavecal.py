import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import numpy as np
from astropy.io import fits as fitsio
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy import interpolate

from .utils import NGaussFunc, minmax


def linerise_wavelength(rawimagedata, wavecalfile="cal_files/wavesoln.npz"):
    npzfile = np.load(wavecalfile)
    wavecal = npzfile["wavecal"]
    newwave = npzfile["newwave"]

    # min and max wave in all cols.
    minwave = wavecal.min(axis=1).max()
    maxwave = wavecal.max(axis=1).min()
    minwavedelta = np.diff(wavecal, axis=1).min()

    newwave = np.arange(minwave, maxwave, minwavedelta)

    interpimg = np.zeros((wavecal.shape[0], newwave.shape[0]))

    for col in range(0, wavecal.shape[0]):
        f = interpolate.interp1d(wavecal[col, :], rawimagedata[col, :])
        interpimg[col, :] = f(newwave)

    return interpimg


def fit_arc_lines(arcimg, spatial_col_skip=1, wavecalfile=""):
    arcimg = arcimg * 1.0 / np.max(arcimg, axis=1)[:, None]
    spatialaxis = np.arange(0, arcimg.shape[0])
    waveaxis = np.arange(0, arcimg.shape[1])

    arcgausfits = np.zeros((len(range(1, arcimg.shape[0], spatial_col_skip)) + 1, 28))

    spec = arcimg[0, :]
    peaks, properties = find_peaks(spec, height=0.01, width=1.5, prominence=0.01)

    y0 = np.zeros((peaks.size * 3))
    y0[0::3] = peaks
    y0[1::3] = properties["peak_heights"]
    y0[2::3] = properties["widths"] * 0.5
    y0 = np.append(y0, 0.02)

    arcgausfits[0, :], pcov = curve_fit(NGaussFunc, waveaxis, spec, p0=y0)
    i = 0
    print("Fitting Arc Lines in each col...")
    for col in tqdm(range(1, arcimg.shape[0], spatial_col_skip)):
        i += 1
        spec = arcimg[col, :]
        y0 = arcgausfits[i - 1, :]
        arcgausfits[i, :], pcov = curve_fit(NGaussFunc, waveaxis, spec, p0=y0)

    pos = arcgausfits[:, range(0, arcgausfits.shape[1] - 1, 3)]
    peakheight = arcgausfits[:, range(0, arcgausfits.shape[1] - 1, 3)]
    width = arcgausfits[:, range(2, arcgausfits.shape[1] - 1, 3)]

    smoothed_pos = np.zeros_like(pos)

    for row in tqdm(range(0, pos.shape[1])):
        smoothed_pos[:, row] = savgol_filter(pos[:, row], 21, 3)

    spectral_lines = (
        np.asarray(
            [
                4358.328,
                5460.735,
                5769.598,
                5790.663,
                6965.4307,
                7067.2175,
                7272.9359,
                7383.9805,
                7503.8691,
            ]
        )
        / 10
    )

    wavecal = np.zeros((smoothed_pos.shape[0], waveaxis.shape[0]))
    p = []
    for i in range(0, smoothed_pos.shape[0]):
        z = np.polyfit(smoothed_pos[i, :], spectral_lines, 4)
        p.append(np.poly1d(z))
        wavecal[i, :] = p[-1](waveaxis)

    minwave = wavecal.min(axis=1).max()
    maxwave = wavecal.max(axis=1).min()
    minwavedelta = np.diff(wavecal, axis=1).min()

    newwave = np.arange(minwave, maxwave, minwavedelta)

    interpimg = np.zeros((wavecal.shape[0], newwave.shape[0]))

    for col in range(0, wavecal.shape[0]):
        f = interpolate.interp1d(wavecal[col, :], arcimg[col, :])
    interpimg[col, :] = f(newwave)

    result = {
        "wavecal": wavecal,
        "pos": pos,
        "peakheight": peakheight,
        "width": width,
        "smoothed_pos": smoothed_pos,
        "newwave": newwave,
    }
    if len(wavecalfile):
        np.savez(wavecalfile, **result)

    return result


if __name__ == "__main__":
    hdulist = fitsio.open("test_files/arc.fits")
    arcimg = np.rot90(hdulist[0].data, -1)
    result = fit_arc_lines(arcimg)
    for key, value in result.items():
        print(key, " : ...")
