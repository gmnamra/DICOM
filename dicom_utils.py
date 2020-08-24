#!/usr/bin/env python3
from sklearn import preprocessing
import numpy as np


def normalize_image (img, norm='l2'):
    return preprocessing.normalize (img, norm)

def normalize_minmax_nan_image (img):
    # """ Normalize image values to [0,1] """
    min_, max_ = float (np.nanmin (img)), float (np.nanmax (img))
    return (img - min_) / (max_ - min_)

def getPixelDataFromDataset(ds):
    """  return the pixel data from the given dataset. If the data
    was deferred, make it deferred again, so that memory is
    preserved. Also applies RescaleSlope and RescaleIntercept
    if available. """

    # Get data
    data = np.array(ds.pixel_array).copy()

    # Obtain slope and offset
    slope = 1
    offset = 0
    needFloats = False
    needApplySlopeOffset = False
    if 'RescaleSlope' in ds:
        needApplySlopeOffset = True
        slope = ds.RescaleSlope
    if 'RescaleIntercept' in ds:
        needApplySlopeOffset = True
        offset = ds.RescaleIntercept
    if int(slope) != slope or int(offset) != offset:
        needFloats = True
    if not needFloats:
        slope, offset = int(slope), int(offset)

    # Apply slope and offset
    if needApplySlopeOffset:

        # Maybe we need to change the datatype?
        if data.dtype in [np.float32, np.float64]:
            pass
        elif needFloats:
            data = data.astype(np.float32)
        else:
            # Determine required range
            minReq, maxReq = data.min(), data.max()
            minReq = min(
                    [minReq, minReq * slope + offset, maxReq * slope + offset])
            maxReq = max(
                    [maxReq, minReq * slope + offset, maxReq * slope + offset])

            # Determine required datatype from that
            dtype = None
            if minReq < 0:
                # Signed integer type
                maxReq = max([-minReq, maxReq])
                if maxReq < 2**7:
                    dtype = np.int8
                elif maxReq < 2**15:
                    dtype = np.int16
                elif maxReq < 2**31:
                    dtype = np.int32
                else:
                    dtype = np.float32
            else:
                # Unsigned integer type
                if maxReq < 2**8:
                    dtype = np.uint8
                elif maxReq < 2**16:
                    dtype = np.uint16
                elif maxReq < 2**32:
                    dtype = np.uint32
                else:
                    dtype = np.float32

            # Change datatype
            if dtype != data.dtype:
                data = data.astype(dtype)

        # Apply slope and offset
        data *= slope
        data += offset

    # Done
    return data



