#!/usr/bin/env python

# import libraries
import pydicom
import os
import sys
import re
import argparse
import glob
from pathlib import Path
import numpy as np
import itertools
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from dicom_utils import getPixelDataFromDataset
from operator import attrgetter
from sklearn import preprocessing
import math

# %%
def normalize_image (img, norm='l2'):
    return preprocessing.normalize (img, norm)




def read_dicom (path):
    """
    INPUTS:
        path:
            a string denoting the path
    OUTPUT:
        list object of dicom files
    """
    # regular expression search for .dcm file
    if re.search (".dcm$", path) is not None:
        return pydicom.dcmread (path, force=True)

def sort_dicom_list (dicom_list):
    """
    INPUTS:
        dicom_list:
            an unsorted list of dicom objects
    OUTPUT:
        sorted list of dicom objects based off of dicom InstanceNumber
    """

    # sort according slice position so we are ALWAYS going from superior -> inferior
    s_dicom_lst = sorted (dicom_list, key=attrgetter ('SliceLocation'), reverse=True)

    return s_dicom_lst


def process_dicom (path, target_x_size=0, target_y_size=0, target_z_size=0):
    """
    At some point we may want to decimate these.
    """
    # initialize rslt_dict
    result_dict = {}

    # read dicom files
    dicom_files = Path (path).glob ('*.dcm')
    files = [x for x in p if x.is_file ()]
    dicom_lst = [read_dicom (x) for x in dicom_files]
    dicom_lst = [x for x in dicom_lst if x is not None]
    # sort list
    dicom_lst = sort_dicom_list (dicom_lst)

    # return image sizes to result dict
    Nx = dicom_lst [0].Rows
    Ny = dicom_lst [0].Columns

    # @todo: decimate using interpolate when we know more
    # perhaps downsample
    # Nz = np.int( dicom_lst[-1].InstanceNumber - dicom_lst[0].InstanceNumber+1)
    Nz = len (dicom_lst)
    target_z_size = Nz
    # also give the option using original image matrix
    target_x_size = Nx
    target_y_size = Ny

    # the following data might not be available due to anonymization
    try:
        result_dict ['patientID'] = dicom_lst [0].PatientID
        result_dict ['AcquisitionDate'] = dicom_lst [0].AcquisitionDate
    except:
        pass

        # get the resolution of the matrix
    scale_x = target_x_size / Nx
    scale_y = target_y_size / Ny
    scale_z = target_z_size / Nz
    result_dict ['image_scale'] = (scale_x, scale_y, scale_z)
    x_sampling = np.float (dicom_lst [0].PixelSpacing [0])
    y_sampling = np.float (dicom_lst [0].PixelSpacing [1])
    z_sampling = np.float (dicom_lst [0].SliceThickness)
    result_dict ['image_resolution'] = (x_sampling * scale_x, y_sampling * scale_y, z_sampling * scale_z)

    # make a list and cast as 3D matrix
    pxl_lst = [x.astype('float32').pixel_array for x in dicom_lst]

    pxl_mtx = pxl_lst
    #pxl_mtx = linear_interpolate (pxl_lst, target_z_size)
    result_dict ['image_data'] = pxl_lst
    return result_dict


# %% the following are modified routines to handle multi-echoes MRI series that
def sort_dicom_list_multiEchoes (dicom_list):
    """
    This function sorts 1st by instance number and then by echo number
    note that echo numbers and echo time correspond
    returns a 2-dimensional list of images with same echoes into one list

    """

    s_dicom_lst = sorted (dicom_list, key=attrgetter ('InstanceNumber'))
    ss_dicom_lst = sorted (s_dicom_lst, key=attrgetter ('EchoNumbers'))
    num_echoes = ss_dicom_lst [-1].EchoNumbers
    dicom_list_groupedby_echoNumber = [None] * num_echoes
    for ii in range (num_echoes):
        tmp_list = []
        for dicomObj in ss_dicom_lst:
            if dicomObj.EchoNumbers == ii + 1:
                tmp_list.append (dicomObj)
        dicom_list_groupedby_echoNumber [ii] = tmp_list

    return dicom_list_groupedby_echoNumber


# %%
def process_dicom_multi_echo (path, target_x_size=0, target_y_size=0, target_z_size=0):
    result_dict = {}
    # store files and append path
    dicom_files = glob.glob (os.path.join (path, "*.dcm"))
    # dicom_files = [path + "/" + x for x in dicom_files]

    # read dicom files
    dicom_lst = [read_dicom (x) for x in dicom_files]
    dicom_lst = [x for x in dicom_lst if x is not None]

    ## Assume all files have same number of echo numbers
    num_echos = int(dicom_lst[0].EchoTrainLength)
    echo_times = []
    for ee in range(num_echos): echo_times.append(ee)
    for dst in dicom_lst:
        en = int(dst.EchoNumbers)
        time = float(dst.EchoTime)
        echo_times [en-1] = time

    result_dict['echo_times'] = echo_times

    # sort list
    # this return a 2-dimension list with all dicom image objects within the same
    # echo number store in the same list
    dicom_lst = sort_dicom_list_multiEchoes (dicom_lst)
    assert(num_echos ==  len (dicom_lst))
    result_dict['num_echos'] = num_echos
    result_dict['num_series'] = len (dicom_lst [1])
    result_dict ['scanning_sequence'] = dicom_lst [0] [0].ScanningSequence
    result_dict ['sequence_variant'] = dicom_lst [0] [0].SequenceVariant
    result_dict ['magnetic_field_strength'] = dicom_lst[0] [0].MagneticFieldStrength
    result_dict ['flip_angle'] = dicom_lst[0] [0].FlipAngle
    result_dict ['TR'] = dicom_lst[0][0].RepetitionTime



    # reports back the first and last instance number of the image sequence
    result_dict ['first_instance_number'] = dicom_lst [0] [0].InstanceNumber
    result_dict ['last_instance_number'] = dicom_lst [-1] [-1].InstanceNumber
    Nimg = np.abs ((result_dict ['last_instance_number'] - result_dict ['first_instance_number']) + 1)
    # return image sizes to result dict
    Nz = 6 #np.int (Nimg / num_echoes)
    Ny = np.int (dicom_lst [0] [0].Rows)
    Nx = np.int (dicom_lst [0] [0].Columns)
    # the following data might not be available due to anonymization
    try:
        result_dict ['patientID'] = dicom_lst [0] [0].PatientID
        result_dict ['AcquisitionDate'] = dicom_lst [0] [0].AcquisitionDate
        result_dict ['ScanningSequence'] = dicom_lst [0][0].ScanningSequence
    except:
        pass
    # make a list and cast as 3D matrix for each echo
    # give the option that don't interpolate along the z-axis if 2-D processing
    if target_z_size == 0:
        target_z_size = Nz
    # also give the option using original image matrix
    if target_x_size == 0:
        target_x_size = Nx

    if target_y_size == 0:
        target_y_size = Ny

    scale_x = target_x_size / Nx
    scale_y = target_y_size / Ny
    scale_z = target_z_size / Nz
    result_dict ['image_scale'] = (scale_x, scale_y, scale_z)
    x_sampling = np.float (dicom_lst [0] [0].PixelSpacing [0])
    y_sampling = np.float (dicom_lst [0] [0].PixelSpacing [1])
    z_sampling = np.float (dicom_lst [0] [0].SliceThickness)
    result_dict ['image_resolution'] = (x_sampling * scale_x, y_sampling * scale_y, z_sampling * scale_z)
    result_dict['Phase'] = []
    result_dict['Magnitude'] = []
    result_dict ['PhaseVoxels'] = []
    result_dict ['MagnitudeVoxels'] = []
    ## We are sorted by echo number.
    ## There are S series each having mag and phase sub series.
    for ii in range (result_dict['num_echos']):
        #@todo use ['ImageType'][2] instead of hardwiring it.
        dst = dicom_lst[ii]
        for ds in dicom_lst[ii]:
            print('%s,%d'%(ds.SeriesTime, ds.EchoNumbers))

        pxl_lst = [getPixelDataFromDataset(ds) for ds in dicom_lst[ii]]
        image_types = [ds['ImageType'][2] for ds in dicom_lst[ii]]
        mtype = [tt == 'M' for tt in image_types]
        ptypes = [tt == 'P' for tt in image_types]
        even = range(0, len(dicom_lst[ii]), 2)
        odd = range(1,len(dicom_lst[ii]), 2)
        phases = np.array ([pxl_lst [e] for e in even])
        mags = np.array ([pxl_lst [o] for o in odd])
        result_dict ['Phase'].append(phases)
        result_dict ['Magnitude'].append (mags)

    result_dict ['water_by_series'] = []  # number of series of number of echos / 2 pairs
    result_dict ['fat_by_series'] = []  # number of series of number of echos / 2 pairs
    result_dict ['average_water_by_series'] = []  # number of series of pairs
    result_dict ['average_fat_by_series'] = []  # number of series of pairs
    result_dict ['opip_by_series'] = []  # number of series of pairs
    result_dict ['pdff'] = []  # number of series of pairs
    ## Number of OP, IP pairs
    opip_pairs = num_echos // 2

    ## we will use average of 2 point dixons in a serie as seed values in per voxel pdff fitings
    ## For each Series create opip_pairs number of dixon2 water and fat images
    # note that we do not divide by 2. It will cancel out when we produce pdff
    #
    for s in range (len(result_dict['Magnitude'])):
        waters = []
        fats = []
        opips = []
        # Collect inphase and out of phase
        for e in range(opip_pairs):
            oIndex = e * 2
            iIndex = oIndex + 1
            op = result_dict['Magnitude'][oIndex][s]
            ip = result_dict['Magnitude'][iIndex][s]
            water = np.add(ip,op)
            fat = np.abs(np.subtract(ip.astype('int16'),op.astype('int16')))
            waters.append(water)
            fats.append(fat)
            opips.append (op)
            opips.append (ip)
        result_dict ['water_by_series'].append (waters)
        result_dict ['fat_by_series'].append (fats)
        ## Produce average water and fet images
        avg_water = np.mean(waters, axis=0)
        avg_fat = np.mean(fats,axis=0)
        result_dict ['average_water_by_series'].append (avg_water)
        result_dict ['average_fat_by_series'].append (avg_fat)
        result_dict['opip_by_series'].append(opips)
        pdff = np.multiply(avg_fat, 100)
        pdff = pdff / np.add(avg_fat, avg_water)
        result_dict ['pdff'].append (pdff)

    # Produce Coarse PDFFs using (IP - OP)/ (IP + IP)
    return result_dict


def show_images (images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len (images) == len (titles)))
    n_images = len (images)
    if titles is None: titles = ['Image (%d)' % i for i in range (1, n_images + 1)]
    fig = plt.figure ()
    for n, (image, title) in enumerate (zip (images, titles)):
        a = fig.add_subplot (cols, np.ceil (n_images / float (cols)), n + 1)
        if image.ndim == 2:
            plt.gray ()
        plt.imshow (image)
        a.set_title (title)
    fig.set_size_inches (np.array (fig.get_size_inches ()) * n_images)
    plt.show ()

# function to display images
def display(images):
    selecs = []
    for col in range(6):
        selecs.append(images[col][5])

    show_images(selecs)

def get_roi_signal(images, roi): # roi is x, y, width, height
    signal = []
    for idx in range(len(images)):
        area = images[idx][roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
        signal.append(np.mean(area))

    return signal

from fitlib import lsqcurvefit

def main_dicom(path):

    results = process_dicom_multi_echo(path)

    show_images (results ['pdff'])
    show_images (results ['opip_by_series'][0])

   # show_images (results['average_water_by_series'])

    loc = [67,128,1,1]
    signal = get_roi_signal(results['opip_by_series'][0], loc)
    print ('Water')
    for ii in results['average_water_by_series']:
        print(ii[loc[1],loc[0]])

    print ('Fat')
    for ii in results ['average_fat_by_series']:
        print (ii [loc [1], loc [0]])

    print(signal)
    plt.plot(signal)
    plt.show()

def fat_peaks_integral (te_seconds):
    dfp = [39,-32,-125,-166,-217,-243]
    drp = [0.047,0.039,0.006,0.120,0.700,0.088]
    ppms = [5.3,4.2,2.75,2.10,1.3,0.9]
    isum = 0
    for i in range(6):
        dd = drp[i] * math.exp(2*math.pi*dfp[i]*te_seconds)
        isum = isum + dd
    return isum

def func(x,tmsec,y):
    s1 = x[0]
    s2 = x[1]
    t2s = x[2]
    out = []
    for tt in tmsec:
        t = tt / 1000
        fpi = fat_peaks_integral(t)
        st = s1 * s1 + s2 * s2 * fpi * fpi + 2 * s1 * s2 * fpi
        st = math.sqrt(st) * math.exp(-t/t2s) * 100
        out.append(st)
    outa = np.array(out)
    ya = np.array(y)
    return outa - ya

from scipy.optimize import least_squares
from scipy.optimize import curve_fit

def func2(x,a, b, c):
    return a * np.exp(-b * x) + c

def main_fit():
    v20 = [15,18,13,17,10,14]
    v10 = [18,20,14,16,11,13]
    v1 = [17.0, 18.0, 16.0, 16.0, 14.0, 14.0]
    pdff_1 = 5.0  # 45, 2.66
    pdff_10 = 6.2  # 30, 2.3
    pdff_20 = 12.5  # 30, 4.3

    tes = [1.2,3.2,5.2,7.2,9.2,11.2]
    e = np.zeros((1,6), dtype=float)
    vsys = 0.020
    water = 0.31
    fat = 0.0033
    signal = v1

    e = func([water,fat,vsys],tes,signal)

    res_lsq = least_squares(func, [vsys, water, fat], args = (tes, signal))
    print (res_lsq)

    xdata = np.linspace(0,4,50)
    y = func2(xdata, 2.5, 1.3, 0.5)
    ydata = y + 0.2 * np.random.normal(size=len(xdata))
    popt, pcov = curve_fit(func2, xdata, ydata)
    print(popt)
    print(pcov)




if __name__ == '__main__':
    if len (sys.argv) < 2: sys.exit(1)
    path = sys.argv [1]
    if not os.path.isdir (path): sys.exit(1)

    main_fit()
    main_dicom (path)

