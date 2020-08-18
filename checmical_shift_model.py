import numpy as np

#!/usr/bin/env python

# import libraries
import pydicom
import os
import re
import glob
from pathlib import Path
import numpy as np
from dicom_utils import getPixelDataFromDataset

from operator import attrgetter
from sklearn import preprocessing


# %%
def normalize_image (img):

    # l2-normalize the samples (rows).
    return preprocessing.normalize (img, norm='l2')
    # """ Normalize image values to [0,1] """
    # min_, max_ = float (np.min (img)), float (np.max (img))
    # return (img - min_) / (max_ - min_)


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

'''
 General DICOM file 
'''
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


'''
    Multi_Echo MRI file. 
    Sorts by instance number and echo number / echo time
'''
def process_dicom_multi_echo (path, target_x_size=0, target_y_size=0, target_z_size=0):
    result_dict = {}
    # store files and append path
    dicom_files = glob.glob (os.path.join (path, "*.dcm"))
    # dicom_files = [path + "/" + x for x in dicom_files]

    # read dicom files
    dicom_lst = [read_dicom (x) for x in dicom_files]
    dicom_lst = [x for x in dicom_lst if x is not None]
    echo_times = {}
    for dst in dicom_lst:
        echo_times[dst.EchoNumber] = dst.EchoTime

    # sort list
    # this return a 2-dimension list with all dicom image objects within the same
    # echo number store in the same list
    dicom_lst = sort_dicom_list_multiEchoes (dicom_lst)
    result_dict['num_echos'] = len (dicom_lst)
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
    for ii in range (result_dict['num_echos']):
        #@todo use ['ImageType'][2] instead of hardwiring it.
        pxl_lst = [normalize_image(getPixelDataFromDataset(ds)) for ds in dicom_lst[ii]]
        image_types = [ds['ImageType'][2] for ds in dicom_lst[ii]]
        mtype = [tt == 'M' for tt in image_types]
        ptypes = [tt == 'P' for tt in image_types]
        even = range(0, len(dicom_lst[ii]), 2)
        odd = range(1,len(dicom_lst[ii]), 2)
        phases = np.array ([pxl_lst [e] for e in even])
        mags = np.array ([pxl_lst [o] for o in odd])
        result_dict ['Phase'].append(phases)
        result_dict ['Magnitude'].append (mags)
        result_dict['PhaseVoxels'].append (np.sum(phases, axis=0))
        result_dict['MagnitudeVoxels'].append(np.sum(mags, axis=0))

    ## Collect magnitude In / Out phase sets. i.e. accross TEs for each series
    ## Create average Ip OP images to generate 2 point dixon water and fat images

    result_dict ['IPOP_mag'] = []
    for s in range (result_dict['num_series']):
        selecs = []
        for e in range(result_dict['num_echos']):
            selecs.append (result_dict['Magnitude'] [e] [s//2])
        result_dict ['IPOP_mag'].append(selecs)

    # Produce Coarse PDFFs using (IP - OP)/ (IP + IP)
    return result_dict


