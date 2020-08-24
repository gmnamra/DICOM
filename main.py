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
import matplotlib.pyplot as plt
from dicom_utils import getPixelDataFromDataset, normalize_minmax_nan_image
from operator import attrgetter
import math
from matplotlib.widgets import EllipseSelector


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
    result_dict['depth'] = num_echos
    result_dict['height'] = np.int (dicom_lst [0] [0].Rows)
    result_dict['width'] = np.int (dicom_lst [0] [0].Columns)
    # the following data might not be available due to anonymization
    try:
        result_dict ['patientID'] = dicom_lst [0] [0].PatientID
        result_dict ['AcquisitionDate'] = dicom_lst [0] [0].AcquisitionDate
        result_dict ['ScanningSequence'] = dicom_lst [0][0].ScanningSequence
    except:
        pass

    scale_x = scale_y = scale_z = 1.0
    result_dict ['image_scale'] = (scale_x, scale_y, scale_z)
    x_sampling = np.float (dicom_lst [0] [0].PixelSpacing [0])
    y_sampling = np.float (dicom_lst [0] [0].PixelSpacing [1])
    z_sampling = np.float (dicom_lst [0] [0].SliceThickness)
    result_dict ['image_resolution'] = (x_sampling * scale_x, y_sampling * scale_y, z_sampling * scale_z)
    result_dict['Phase'] = []
    result_dict['Magnitude'] = []
    result_dict ['MedianPhase'] = []
    result_dict ['MedianMagnitude'] = []
    result_dict ['PhaseVoxels'] = []
    result_dict ['MagnitudeVoxels'] = []
    ## We are sorted by echo number.
    ## There are S series each having mag and phase sub series.
    for ii in range (result_dict['num_echos']):
        #@todo use ['ImageType'][2] instead of hardwiring it.
        mtypes = []
        ptypes = []
        for idx, ds in enumerate(dicom_lst[ii]):
            itype = ds['ImageType'][2]
            #print('%s,%d,%s'%(ds.SeriesTime, ds.EchoNumbers, itype))
            if itype == 'M':mtypes.append(idx)
            elif itype == 'P':ptypes.append(idx)
        pxl_lst = [getPixelDataFromDataset(ds) for ds in dicom_lst[ii]]
        phases = np.array ([pxl_lst [e] for e in ptypes])
        mags = np.array ([pxl_lst [o] for o in mtypes])
        result_dict ['Phase'].append(phases)
        result_dict ['Magnitude'].append (mags)
        medp = np.median(phases, axis=0)
        medm = np.median(mags, axis=0)
        result_dict ['MedianPhase'].append(medp)
        result_dict ['MedianMagnitude'].append (medm)

    ## we will use average of 2 point dixons in a serie as seed values in per voxel pdff fitings
    ## For each Series create opip_pairs number of dixon2 water and fat images
    # note that we do not divide by 2. It will cancel out when we produce pdff
    #
    # Collect inphase and out of phase
    ip1 = result_dict['MedianMagnitude'][0]
    op = result_dict ['MedianMagnitude'] [1]
    ip2 = result_dict ['MedianMagnitude'] [2]
    water = np.add(ip1,op)
    fat = np.subtract(ip1.astype('int16'),op.astype('int16'))
    result_dict ['water'] = water
    result_dict ['fat'] = fat
    pdff = np.multiply(fat, 1000)
    pdff = pdff / np.add(fat, water)
    result_dict ['pdff'] = np.clip(pdff, 0.0, 1000)
    result_dict ['OutOfPhaseMagnitude'] = op
    result_dict ['InPhaseMagnitude'] = ip1
    t2s = np.subtract (ip2.astype ('float'), ip1.astype ('float'))
    s1os2 = np.divide (ip1.astype ('float'), ip2.astype ('float'))
    s1os2 = np.log(s1os2)
    tmp = t2s * s1os2
    result_dict['T2*'] = tmp
    correction = np.exp(np.divide(t2s,result_dict['T2*']))
    ip1_corrected = np.multiply(ip1, correction)
    result_dict['InPhaseMagnitudeCorrected'] = np.clip(ip1_corrected, 0, 255)
    # pdff3 = np.subtract(ip1_corrected,op.astype('float')) / np.add(ip1_corrected,ip1_corrected)
    # pdff3 = np.multiply(pdff3, 100.0)

    # result_dict['pdff3'] = pdff3
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

    return np.array(signal)

def fat_peaks_integral (te_seconds):
    dfp = [39,-32,-125,-166,-217,-243]
    drp = [0.047,0.039,0.006,0.120,0.700,0.088]
    ppms = [5.3,4.2,2.75,2.10,1.3,0.9]
    isum = 0
    for i in range(6):
        dd = drp[i] * math.exp(2*math.pi*dfp[i]*te_seconds)
        isum = isum + dd
    return isum



def generate(x,tmsec):
    s1 = x[0]
    s2 = x[1]
    vsys = x[2]
    v1 = vsys
    v2 = vsys + 1/0.012
    out = []
    w = 2* math.pi / 0.0036
    for tt in tmsec:
        t = tt / 1000
        st = s1 * s1 * math.exp(-2 * v1 * t) + s2 * s2 * math.exp(-2 * v2 * t) + 2 * s1 * s2 * math.exp(- v1 * t - v2 * t)*math.cos(w * t)
        st = math.sqrt(st)
        out.append(st)
    return np.array(out)

def func(x,tmsec,y):
    return generate(x,tmsec) - y

from scipy.optimize import least_squares
from scipy.optimize import curve_fit

def func2(x,a, b, c):
    return a * np.exp(-b * x) + c


def signal_fit(signal, water,fat, t2r):
    tes = [1.2,3.2,5.2,7.2,9.2,11.2]
    e = np.zeros((1,6), dtype=float)

    res_lsq = least_squares(func, [water, fat, t2r], method='lm', args = (tes, signal))
    e = generate(res_lsq.x, tes)
    return res_lsq, e

def curve_fit_example():
    xdata = np.linspace(0,4,50)
    y = func2(xdata, 2.5, 1.3, 0.5)
    ydata = y + 0.2 * np.random.normal(size=len(xdata))
    popt, pcov = curve_fit(func2, xdata, ydata)
    print(popt)
    print(pcov)



def main_dicom(path):

    results = process_dicom_multi_echo (path)
    id_str = Path (path).stem

    def onselect (eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        print ('startposition: (%f, %f)' % (eclick.xdata, eclick.ydata))
        print ('endposition  : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print ('used button  : ', eclick.button)
        ctr = [int((eclick.xdata + erelease.xdata) / 2.0), int((eclick.ydata + erelease.ydata) / 2.0)]
        patch = [int(math.fabs(eclick.xdata - erelease.xdata) / 2.0) , int(math.fabs(eclick.ydata - erelease.ydata) / 2.0)]
        loc = [ctr[0],ctr[1],patch[0],patch[1]]
        insr = run_location(loc, results)
        plot_instant_result (insr)

    def toggle_selector (event):
        print (' Key pressed.')
        if event.key in ['Q', 'q'] and toggle_selector.ES.active:
            print ('EllipseSelector deactivated.')
            toggle_selector.RS.set_active (False)
        if event.key in ['A', 'a'] and not toggle_selector.ES.active:
            print ('EllipseSelector activated.')
            toggle_selector.ES.set_active (True)



    def run_location(loc, results):
        signal = get_roi_signal(results['MedianMagnitude'], loc)
        print ('Water')
        ii = results['water']
        pwater = ii[loc[1],loc[0]]
        ii = results ['fat']
        pfat = ii [loc [1], loc [0]]
        ii = results ['pdff']
        pdff = ii [loc [1], loc [0]] / 10
        ii = results ['T2*']
        print('T2* %f'%(ii[loc[1], loc[0]]))

        res_lsq, e = signal_fit(signal / 100, pwater / 100, pfat / 100, 0.0)
        print(res_lsq)
        print(e)
        hist, edges = np.histogram(results['pdff'], bins=range(500))
        fitted = res_lsq.x
        fitted_water = math.fabs(fitted[0]) * 100
        fitted_fat = math.fabs(fitted[1]) * 100
        fitted_pdff = (fitted_fat) * 100 / (fitted_fat + fitted_water)

        output = "{id} \n @ ({x},{y} path_size = {ps}]\n \n Signal Based: \n\n [{pw:2.2f},{pf:2.2f}] -> {pff:2.2f} \n \n Model Based \n\n[{epw:2.2f},{epf:2.2f}] -> {epff:2.2f} ".format \
            (id=id_str, x=loc[0],y=loc[1], ps=loc[3], pw=pwater, pf=pfat,pff=pdff, epw= fitted_water, epf= fitted_fat, epff=fitted_pdff)
        print (output)
        instant_results = {}
        instant_results['results'] = results
        instant_results['signal'] = signal
        instant_results['e'] = e
        instant_results['hist'] = hist
        instant_results['output'] = output
        instant_results['loc'] = loc

        return instant_results


    class Formatter (object):
        def __init__ (self, im):
            self.im = im

        def __call__ (self, x, y):
            z = self.im.get_array () [int (y), int (x)]
            return 'x={:.01f}, y={:.01f}, z={:.01f}'.format (x, y, z)

    def plot_instant_result(instant_results):
        results = instant_results['results']
        signal = instant_results ['signal']
        e = instant_results ['e']
        hist = instant_results ['hist']
        output = instant_results ['output']
        roi = instant_results['loc']
        patch = results['pdff'][roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]

        fig, axs = plt.subplots(2, 4, figsize=(20, 10), frameon=False,
                              subplot_kw={'xticks': [], 'yticks': []})
        axs[0, 0].imshow(results ['MedianMagnitude'][0], cmap='gray')
        axs[0, 0].set_title('Median IP')
        axs [0, 1].imshow (results ['MedianMagnitude'] [1], cmap='gray')
        axs[0, 1].set_title('Median OP')
        im = axs[0, 2].imshow (results ['pdff'], interpolation='none', cmap='gray')
        axs[0, 2].format_coord = Formatter (im)
        axs [0, 2].set_title ('PDFF')
        axs[0, 3].imshow(patch, cmap='gray')
        axs [0, 3].set_title ('Selected Patch')

        axs[1, 0].plot(signal)
        axs[1, 0].set_title('Voxel')
        axs[1, 1].plot(e)
        axs [1, 1].set_title ('Fitted')

        x = range(499)
        axs[1, 2].plot(x,hist)
        axs[1,2].ticklabel_format (axis='y', style='scientific', scilimits=(0, 0))

        def fine2coarse(x):
            return x // 10
        def coarse2fine(x):
            return x * 10

        axs[1, 2].set_title ('pdff histogram')
        secax = axs[1,2].secondary_xaxis ('top', functions=(fine2coarse, coarse2fine))
        secax.set_xlabel ('hist [%]')

        axs [1, 3].text (0.5, 0.75, output, verticalalignment='top', horizontalalignment='center',
                         transform=axs [1, 3].transAxes,
                         color='green', fontsize=15)
        toggle_selector.ES = EllipseSelector (axs[0, 2], onselect, drawtype='line', lineprops=dict(color="red", linestyle="-", linewidth=2, alpha=0.5))
        fig.canvas.mpl_connect ('key_press_event', toggle_selector)

        plt.autoscale
        plt.show()

    instant_res = run_location ([75, 100, 8, 8], results)
    plot_instant_result(instant_res)


if __name__ == '__main__':
    if len (sys.argv) < 2: sys.exit(1)
    path = sys.argv [1]
    if not os.path.isdir (path): sys.exit(1)

    main_dicom (path)

