import pydicom
import cv2 as cv
import PIL
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy import matrix
from numpy import linalg
import os
import sys
import math
import struct
import itertools
from dicom_utils import getPixelDataFromDataset


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


# function to display images
def display(images, titles=['']):
    ilen = len(images)
    tlen = len(titles)
    if ilen == 0 or ilen != tlen: return False

    if isinstance(images[0], list):
        c = len(images[0])
        r = len(images)
        images = list(itertools.chain(*images))
    else:
        c = len(images)
        r = 1
    plt.figure(figsize=(4 * c, 4 * r))
    gs1 = gridspec.GridSpec(r, c, wspace=0, hspace=0)
    # gs1.update(wspace=0.01, hspace=0.01) # set the spacing between axes.
    # titles = itertools.cycle(titles)
    for i in range(r * c):
        im = images[i]
        title = titles[i]
        plt.subplot(gs1[i])
        # Don't let imshow doe any interpolation
        plt.imshow(im, cmap='gray', interpolation='none')
        plt.axis('off')
        if i < c:
            plt.title(title)
    plt.tight_layout()
    plt.show()
    return True


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


def normalize_image (img):
    """ Normalize image values to [0,1] """
    min_, max_ = float (np.min (img)), float (np.max (img))
    return (img - min_) / (max_ - min_)



def load_scan(folder):
    assert(os.path.isdir(folder))
    p = Path (folder).glob ('*.dcm')
    files = [x for x in p if x.is_file ()]
    files.sort(key=lambda f: int(str(f.name).split('.')[9]))
    slices = [pydicom.dcmread (str(file)) for file in files]
    params = []
    pparams = []
    def print_get(data,  stringKey):
        val = _get(data, stringKey)
     #   if not ( val is None):
    #        print(('[%s] = %f') % (stringKey, val))
        return val

    series = {}
    for idx, slice in enumerate(slices):
        # print('---%d----' % (idx))
        # print_get(slice,'EchoTime')
        sn = (int)(print_get(slice,'SeriesNumber'))
        if not ( sn in series):
            series[sn] = []
        series[sn].append(idx)
        # print('---%d----' % (idx))

    stacks = {}
    for key,value in series.items():
  #       print('%d, %d' % (key, len(value)))
         stacks[key] = np.stack ([normalize_image(getPixelDataFromDataset (slices[s])) for s in value])

    avgs = []
    titles = []
    arrays3d = []
    for idx, stk in enumerate(stacks):
        avg_ = stacks[stk].mean(axis=0)
        avgs.append(avg_)
        titles.append(str(idx))
        arrays3d.append(np.array(stacks[stk]))

   # display (avgs, titles)
    shape = avgs[0].shape
    pdff = np.zeros((shape), dtype='float')

    for j in range(shape[0]):
        for i in range(shape[1]):
            ip = arrays3d [0] [:, j, i]
            op = arrays3d [1] [:, j, i]
            if np.all(op > 0.0) and np.all(ip > 0.0):
                # if i%j == 0:
                #     f, axs = plt.subplots (1, 2, figsize=(20, 10), frameon=False,
                #                            subplot_kw={'xticks': [], 'yticks': []})
                #     axs [0].plot (range(7), ip)
                #     axs [1].plot (range(7), op)
                #     plt.show()
                pdff[j][i] = np.mean(ip)


    pdff = normalize_image(pdff)
    pdff = pdff * 100

    np.histogram (pdff, bins=range(100))
    hist, bins = np.histogram (pdff, bins=range(100), density=True)
    print(hist)
    print(bins)
    print(np.median(pdff))
    # f, axs = plt.subplots (1, 3, figsize=(20, 10), frameon=False,
    #                        subplot_kw={'xticks': [], 'yticks': []})
    # axs [0].imshow(pdff, cmap='gray')
    plt.hist (pdff, bins=range(100))
    plt.title ("histogram")
    plt.show ()

    return slices


def main():
    #slices = load_scan ('/Volumes/t5backup/MRIp4/mri_images/ideal_images/1_IDEAL_20254/')
    slices = load_scan ('/Volumes/t5backup 1/MRIp4/mri_images/shmolli_images/20SHMOLLI_20204')
    slices = load_scan ('/Volumes/t5backup 1/MRIp4/mri_images/shmolli_images/10SHMOLLI_20204')
    slices = load_scan ('/Volumes/t5backup 1/MRIp4/mri_images/shmolli_images/1SHMOLLI_20204')

    print ('done')


if __name__ == '__main__':
    main ()

