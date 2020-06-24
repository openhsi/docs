import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.signal import savgol_filter


def func(x, *params):  # x0 pk width
    y = np.zeros_like(x)
    for i in range(0, len(params) - 1, 3):
        ctr = params[i]
        amp = params[i + 1]
        wid = params[i + 2]
        y = y + amp * np.exp(-(((x - ctr) / wid) ** 2))
    return y + params[-1]


hdulist = pyfits.open("arc.fits")

arcimg = np.rot90(hdulist[0].data, -1)
arcimg = arcimg * 1.0 / np.max(arcimg, axis=1)[:, None]

plt.figure(figsize=(5, 5))
imgplot = plt.imshow(np.log10(arcimg))

x = np.arange(0, arcimg.shape[1])

fits = []

for row in range(0, arcimg.shape[0]):
    spec = arcimg[row, :]
    peaks, properties = find_peaks(spec, height=0.025, width=1)

    y0 = np.zeros((peaks.size * 3))
    y0[0::3] = peaks
    y0[1::3] = properties["peak_heights"]
    y0[2::3] = properties["widths"] * 0.5
    y0 = np.append(y0, 0.02)

    popt, pcov = curve_fit(func, x, spec, p0=y0)
    fits.append(popt)
    # fit = func(x, *popt)

np.save("fits.npy", np.asarray(fits))
