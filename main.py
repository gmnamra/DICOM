import pydicom
import cv2 as cv
import PIL
import matplotlib.pyplot as plt
import numpy as np
from numpy import matrix
from numpy import linalg
import os
import sys
import math
import struct



# common packages
import numpy as np
import os
import copy
from math import *
import matplotlib.pyplot as plt
from functools import reduce
# reading in dicom files
import pydicom as dicom
from pydicom.datadict import tag_for_keyword, keyword_for_tag, repeater_has_keyword

# skimage image processing packages
from skimage import measure, morphology
from skimage.morphology import ball, binary_closing
from skimage.measure import label, regionprops
# scipy linear algebra functions
from scipy.linalg import norm
import scipy.ndimage
# ipywidgets for some interactive plots
from ipywidgets.widgets import *
import ipywidgets as widgets
# plotly 3D interactive graphs
import plotly
from plotly.graph_objs import *
import chart_studio.plotly as py
from pathlib import Path
# set plotly credentials here
# this allows you to send results to your account plotly.tools.set_credentials_file(username=your_username, api_key=your_key)


# These are DICOM standard tags
TAG_SOP_CLASS_UID = (0x0008, 0x0016)

# These are some Siemens-specific tags
TAG_CONTENT_TYPE                          = (0x0029, 0x1008)
TAG_SPECTROSCOPY_DATA_DICOM_SOP           = (0x5600, 0x0020)
TAG_SPECTROSCOPY_DATA_SIEMENS_PROPRIETARY = (0x7fe1, 0x1010)


firstFile = r'/Volumes/t5backup/MRIp4/mri_images/shmolli_images/1SHMOLLI_20204/1.3.12.2.1107.5.2.18.41754.201609121313103844008230.dcm'
secondFile = r'/Volumes/t5backup/MRIp4/mri_images/ideal_images/1_IDEAL_20254/1.3.12.2.1107.5.2.18.41754.2016091213121636910507776.dcm'


def _empty_string(value, default):
    if value == '':
        return default
    else:
        return value


def _get (dataset, tag, default=None):
    """Returns the value of a dataset tag, or the default if the tag isn't
    in the dataset.
    PyDicom datasets already have a .get() method, but it returns a
    dicom.DataElement object. In practice it's awkward to call dataset.get()
    and then figure out if the result is the default or a DataElement,
    and if it is the latter _get the .value attribute. This function allows
    me to avoid all that mess.
    It is also a workaround for this bug (which I submitted) which should be
    fixed in PyDicom > 0.9.3:
    http://code.google.com/p/pydicom/issues/detail?id=72
    Also for this bug (which I submitted) which should be
    fixed in PyDicom > 0.9.4-1:
    http://code.google.com/p/pydicom/issues/detail?id=88

    bjs - added the option that the tag may exist, but be blank. In this case
          we will return the default value. This is especially important if
          the data has been run through an anonymizer as many of these leave
          the tag but set it to a blank string.

    """
    if tag not in dataset:
        return default

    if dataset [tag].value == '':
        return default

    return dataset [tag].value




def load_scan(folder):
    assert(os.path.isdir(folder))
    p = Path (folder).glob ('*.dcm')
    files = [x for x in p if x.is_file ()]
    files.sort(key=lambda f: int(str(f.name).split('.')[9]))
    slices = [pydicom.dcmread (str(file)) for file in files]
    image = np.stack ([s.pixel_array for s in slices])
    image = np.array (image, dtype=np.int16)
    foo = np.histogram(image, bins=256)
    params = []
    pparams = []
    def print_get(data,  stringKey):
        val = _get(data, stringKey)
        if not ( val is None):
            print(('[%s] = %f') % (stringKey, val))
        return val

    series = {}
    for idx, slice in enumerate(slices):
        print('---%d----' % (idx))
        print_get(slice,'EchoTime')
        sn = (int)(print_get(slice,'SeriesNumber'))
        if not ( sn in series):
            series[sn] = []
        series[sn].append(idx)
        print('---%d----' % (idx))

    stacks = {}
    for key,value in series.items():
         print(len(value))
         stacks[key] = np.stack ([slices[s].pixel_array for s in value])

    avg_41 = stacks[41].mean(axis=0)
    plt.imshow(avg_41)
    plt.show()

    return slices


def main():
#    slices = load_scan ('/Volumes/t5backup/MRIp4/mri_images/ideal_images/1_IDEAL_20254/')
    slices = load_scan ('/Volumes/t5backup 1/MRIp4/mri_images/shmolli_images/1SHMOLLI_20204')

    # slice = pydicom.dcmread(firstFile)
    print ('done')


if __name__ == '__main__':
    main ()

