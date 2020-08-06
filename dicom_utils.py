#!/usr/bin/env python3

import numpy as np
import dicom
import sys

def get_image(ds_open_file):
    """Returns a numpy array containing the actual image data of the image,
    in ascending monochrome (high = white), known in DICOM as MONOCHROME2"""

    ds = ds_open_file

    # Photometric Interp. tells us how to convert to MONOCHROME2
    # DICOM standard says this value MUST exist at [0x28,0x4]
    # See http://www.medicalconnections.co.uk/kb/Photometric_Interpretations
    pminterp = ds[0x28,0x4].value
    pixels = ds.pixel_array

    ## MONOCHROME2 is what we want. Do a sanity check and exit
    if pminterp == 'MONOCHROME2':
        if len(pixels.shape) != 2:
            raise ValueError("DICOM claims that this image " +
                             "is grayscale but that's doesn't seem right.")
        return pixels

    ## MONOCHROME1 is almost right, but we need to invert the pixels
    elif pminterp == 'MONOCHROME1':
        if len(pixels.shape) != 2:
            raise ValueError("DICOM claims that this image " +
                             "is grayscale but that's doesn't seem right.")

        ## Need to reverse the values for white-hot (instead of black-hot)
        if "BitsStored" in a:
            maxval = 2**a.BitsStored - 1
        else:
            print("[WARN] Could not find Bits Stored field. Assuming 12-bit pixels")
            maxval = 4095 #A sane default (12-bit scanner)

        # Invert image: set 0 -> max, 1 -> max - 1, etc.
        for x in np.nditer(pixels, op_flags=['readwrite']):
            x[...] = maxval - x # nditer rules

        return pixels

    ## For RGB, we need to convert down to monochrome.
    elif pminterp == 'RGB':
        # We use the default values provided by DCMTK for this: red = 0.299, green = 0.587, blue = 0.114
        # See http://support.dcmtk.org/docs/classDicomImage.html#80077b29cef7b9dbbe21dc19326416ad

        if len(pixels.shape) != 3 or pixels.shape[2] != 3:
            raise ValueError("DICOM claims that this image " +
                             "is RGB but that's doesn't seem right.")

        colorweights = [0.299, 0.587, 0.114]
        output = np.mean(pixels * colorweights, axis=2) #This line may be suspect

        return output

    ## How is this encoded??
    else:
        raise ValueError('I tried to get the image data for this image, '
                         +'but its PhotometricInterpretation value is: ' + pminterp
                         +'which I do not understand.')

def get_image_metadata(ds_open_file):
    """Given an open file, gets only the metadata  by deleting the data"""

    del ds_open_file.PixelData
    return ds_open_file

def get_value(dicoms,valstring):
    """Given a set of DICOM images, searches for valstring in the DICOM data.
    Returns the first found value among dicoms."""

    for dicom in dicoms:
        if valstring in dicom:
            return dicom.data_element(valstring).value

    # Didn't find it
    print("Could not find tag " + valstring)
    return None

def get_series_metadata(series):
    """Given a collection of DICOM DataSets that form a series,
    retrieves interesting series metadata"""
    smetadata = dict()

    data_names = ['SeriesDate', 'SeriesTime', 'SeriesDescription', 'SliceThickness', 'KVP',
                  'PhotometricInterpretation', 'ImageOrientationPatient' ]

    for data_name in data_names:
        smetadata[data_name] = get_value(series, data_name)

    # Add dimensions of series
    smetadata['x_ct'] = get_value(series,'Rows')
    smetadata['y_ct'] = get_value(series,'Columns')
    smetadata['z_ct'] = len(series)

    return smetadata

def get_study_metadata(study):
    """Given a collection of DICOM DataSets that form a study,
    retrieves interesting study metadata"""
    smetadata = dict()

    data_names = ['StudyDate', 'StudyTime', 'StudyDescription', 'PatientAge', 'PatientBirthDate',
                  'PatientID', 'PatientSex']

    for data_name in data_names:
        smetadata[data_name] = get_value(study, data_name)

    return smetadata

def get_dicom_file(filename):
    return dicom.read_file(filename)

def get_hounsfield_coeffs(ds_open_file):
    """Returns the slope-intercept pair needed to convert an image
    to Hounsfield units"""
    rescale_slope = ds_open_file[0x28, 0x1053].value
    rescale_intercept = ds_open_file[0x28,0x1052].value

    return (rescale_slope,rescale_intercept)


