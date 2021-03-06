# AUTOGENERATED! DO NOT EDIT! File to edit: 01_calibrate.ipynb (unless otherwise specified).

__all__ = ['sum_gaussians', 'fit_arc_lines2']

# Cell
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy import interpolate
from fastprogress.fastprogress import master_bar, progress_bar
import h5py

# Cell

def sum_gaussians(x:"wavelength np.array",
                    *args:"amplitude, peak position, peak width, constant") -> np.array:
    split = len(args)//3
    A   = args[0:split]         # amplitude
    mu  = args[split:2*split]   # peak position
    sigma = args[split*2:-1]    # peak stdev
    c   = args[-1]              # offset
    return np.array( [A[i] * np.exp( - np.square( (x - mu[i])/sigma[i] ) )
                        for i in range(len(A))] ).sum(axis=0) + c


def fit_arc_lines2(arc_file:str = "cal_files/arc.hdf5", wave_save_file:str = None, skip:int = 1,
                    show:bool = True) -> dict:
    """Fit a bunch of guassians on top of a spectrum. ???"""
    with h5py.File(arc_file, "r") as f:
        arc_img = np.array(f['arc_img'],dtype=np.float64)

        # normalise the image?
        arc_img /= np.max(arc_img, axis = 1)[:, None]

        # rows -> spatial axis, cols -> wavelength axis
        rows, cols = arc_img.shape
        x_array, wavelengths = np.arange(rows), np.arange(cols)

        # init arrays. why 28 cols?
        # took a while, but this is rows*(9peaks*3arrays+1constant)
        # the assumption here is that there will only be 9 peaks.
        arc_gauss_fit = np.zeros((rows//skip,28))

        # init with first pixel's spectrum
        # does spec mean species?
        spec    = arc_img[0,:]
        mu, props = find_peaks(spec, height = 0.01, width = 1.5, prominence = 0.01)
        A       = props["peak_heights"]
        sigma   = 0.5 * props["widths"]
        c       = 0.02
        params0 = [*A,*mu,*sigma,c]



        if show:
            plt.subplots(nrows=1,ncols=2,figsize=(10,5))
            plt.subplot(1,2,1)
            plt.plot(wavelengths,spec)
            plt.plot(mu, A, 'r*')
            plt.xlabel('wavelength (nm) with some offset ???')
            plt.ylabel('normalised amplitude')
            #plt.show()

        #breakpoint()
        #arc_gauss_fit[0,:], _ = curve_fit(sum_gaussians, wavelengths, spec, p0=params)
        print('Fit arc lines for each spatial pixel')
        #skip = rows
        for i in progress_bar(range(0,rows,skip)):
            params = params0 if i == 0 else arc_gauss_fit[i-1,:]
            arc_gauss_fit[i,:], _ = curve_fit(sum_gaussians, wavelengths, arc_img[i,:], p0=params)

        split = len(params0)//3
        A     = arc_gauss_fit[:,:split]
        mu    = arc_gauss_fit[:,split:2*split]
        sigma = arc_gauss_fit[:,2*split:-1]

        # why smooth the peak centres?
        # shape is (spatial pixels,9 peaks found)
        smooth_mu = np.zeros_like(mu)
        for j in range(split):
            smooth_mu[:,j] = savgol_filter(mu[:,j], 21, 3)

        spectral_lines = np.asarray([4358.328, 5460.735, 5769.598, 5790.663, 6965.4307,
                                7067.2175, 7272.9359, 7383.9805, 7503.8691]) / 10

        wave_cal = np.zeros((rows,cols))

        poly_funcs = [np.poly1d( np.polyfit(smooth_mu[i,:], spectral_lines, 4) ) for i in range(rows)]
        #breakpoint()
        wave_cal = np.array([p(wavelengths) for p in poly_funcs])

        # what is the reasoning behind .max() after min()? and vice versa
        min_wavelength = wave_cal.min(axis=1).max()
        max_wavelength = wave_cal.max(axis=1).min()
        delta_wavelength = np.diff(wave_cal, axis=1).min()

        #breakpoint()
        newwave = np.arange(min_wavelength, max_wavelength, delta_wavelength)
        interp_img = np.zeros((rows,len(newwave)))

        f = [interpolate.interp1d(wave_cal[i,:], arc_img[i,:]) for i in range(rows)]
        for i in range(rows):
            interp_img[i,:] = f[i](newwave)

        if show:
            plt.subplot(1,2,2)
            plt.imshow(wave_cal)
            plt.xlabel('wavelength (nm) with some offset ???')
            plt.ylabel('spatial pixels')
            plt.colorbar()
            plt.show()

        result = {  "wavecal": wave_cal,
                    "pos": mu,
                    "peakheight": A,
                    "width": sigma,
                    "smoothed_pos": smooth_mu,
                    "newwave": newwave}

        if wave_save_file:
            with h5py.File(wave_save_file, "w") as f:
                for k, v in result.items():
                    f.create_dataset(k, data=np.array(v, dtype=np.float64))

        return result

