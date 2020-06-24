import numpy as np
import pyfits
from ximea import xiapi

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print("Opening first camera...")
cam.open_device()

# settings
cam.set_exposure(5000)
print("Exposure was set to %i us" % cam.get_exposure())

cam.set_imgdataformat("XI_RAW16")
cam.set_output_bit_depth("XI_BPP_12")
cam.enable_output_bit_packing()
cam.disable_aeag()

cam.set_binning_vertical(2)
cam.set_binning_vertical_mode("XI_BIN_MODE_SUM")

cam.set_width(896)
cam.set_offsetX(528)
# cam.set_height(1544)
# cam.set_offsetY(0)

# cam.disable_auto_wb()

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisition
print("Starting data acquisition...")
cam.start_acquisition()

data = np.zeros((cam.get_height(), cam.get_width(), 100), dtype=np.uint16)

for i in range(100):
    # get data and pass them from camera to img
    cam.get_image(img)

    # get raw data from camera
    # for Python2.x function returns string
    # for Python3.x function returns bytes
    data[:, :, i] = img.get_image_data_numpy()

    # transform data to list
    # data = list(data_raw)

    # print image data and metadata
    print("Image number: " + str(i))
    print("\n")

pyfits.writeto("arc.fits".format(i), np.mean(data, axis=2), clobber=True)

# stop data acquisition
print("Stopping acquisition...")
cam.stop_acquisition()

# stop communication
cam.close_device()

print("Done.")
