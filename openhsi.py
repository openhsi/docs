import numpy as np
import pyfits
from ximea import xiapi


class openhsi(object):
    def __init__(self, serialnumber="", xbinwidth=896, xbinoffset=528):
        self.xicam = xiapi.Camera()
        if len(serialnumber) == 0:
            self.xicam.open_device()
        else:
            self.xicam.open_device_by_SN(serialnumber)

        self.exposure = 5
        self.gain = 0

        self.xicam.set_imgdataformat("XI_RAW16")
        self.xicam.set_output_bit_depth("XI_BPP_12")
        self.xicam.enable_output_bit_packing()
        self.xicam.disable_aeag()

        self.xicam.set_binning_vertical(2)
        self.xicam.set_binning_vertical_mode("XI_BIN_MODE_SUM")

        self.xbinwidth = 896
        self.xbinoffset = 528

        self.img = xiapi.Image()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.xicam.stop_acquisition()
        self.xicam.close_device()

    def start(self):
        self.xicam.start_acquisition()

    def stop(self):
        self.xicam.stop_acquisition()

    def get_single_image(self):
        self.xicam.get_image(self.img)
        return np.rot90(self.img.get_image_data_numpy(), -1)

    def get_image_series(self, numimages=100):
        data = np.zeros(
            (self.xicam.get_width(), self.xicam.get_height(), numimages),
            dtype=np.uint16,
        )
        for i in range(numimages):
            self.xicam.get_image(self.img)
            data[:, :, i] = np.rot90(self.img.get_image_data_numpy(), -1)
        return data

    def imgsize(self):
        return self.xicam.get_height(), self.xicam.get_width()

    @property
    def exposure(self):
        """current exposure property ms."""
        return self.xicam.get_exposure() / 1000

    @exposure.setter
    def exposure(self, val):
        self.xicam.set_exposure_direct(val * 1000)

    @property
    def gain(self):
        """current exposure property. 0 to 24 dB"""
        return self.xicam.get_gain()

    @exposure.setter
    def gain(self, val):
        self.xicam.set_gain_direct(val)

    @property
    def xbinwidth(self):
        """ """
        return self.xicam.get_width()

    @exposure.setter
    def xbinwidth(self, val):
        self.xicam.set_width(val)

    @property
    def xbinoffset(self):
        """ """
        return self.xicam.get_offsetX()

    @exposure.setter
    def xbinoffset(self, val):
        self.xicam.set_offsetX(val)
