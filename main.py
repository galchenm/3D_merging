#!/usr/bin/env python

import argparse
import sys

import matplotlib.pyplot as plt
import os

import cfel_geom as cg
import geometry_funcs as gf

import numpy as np
import numpy.linalg as ln
import ctypes as ct
from scipy import stats, ndimage, misc

from struct import *
import re

import h5py as h5 


MASK_GOOD = 1
MASK_BAD = 0

MinVal=1e-10


def _np_ptr(np_array):
    return ct.c_void_p(np_array.ctypes.data)

def pSubLocalBG(inpAr, stx, enx, sty, eny, fNumX, radX, radY, badVal, smoothedAr):
    inpAr = np.array(inpAr, dtype=np.float32)
    lib = ct.CDLL( '/home/galchenm/bin/main_prog/SubLocalBG.so' )
    pfun = lib.SubLocalBG # bool SubLocalBG(float* inpAr, int stx, int enx, int sty, int eny, int fNumX, int radX, int radY, float badVal, float* smoothedAr)
    # pfun.restype = ct.c_bool
    pfun.restype = ct.c_int
    # pfun.argtypes = (ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.POINTER(ct.c_float))
    pfun.argtypes = (ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_void_p)
    flag = pfun(_np_ptr(inpAr), stx, enx, sty, eny, fNumX, radX, radY, badVal, smoothedAr)
    # print(inpAr[490])
    return flag, inpAr

def pSubtractBgLoopMedianCutoff(data, pix_nn, pix_r, hitfinderMinSNR, ADCthresh, bstpReg, radialcurve, cutoff, curveSmooth):
    #float SubtractBgLoopMedianCutoff(float* data, long pix_nn, int *pix_r, float hitfinderMinSNR, float ADCthresh, float bstpReg, float* radialcurve, float cutoff, int curveSmooth)
    print("--------SubtractBgLoopMedianCutoff--------\n")
    data = np.array(data, dtype = np.float32)
    pix_r = np.array(pix_r, dtype=np.int32)
    len_radialcurve = max(pix_r)
    print(len_radialcurve)
    radialcurve = np.array([0]*len_radialcurve, dtype=np.float32)
    print("radialcurve: ",radialcurve)
    lib = ct.CDLL( '/home/galchenm/bin/main_prog/SubLocalBG.so')
    pfun = lib.SubtractBgLoopMedianCutoff
    pfun.restype = ct.c_float
    pfun.argtypes = (ct.c_void_p, ct.c_long, ct.c_void_p, ct.c_float, ct.c_float, ct.c_float, ct.c_void_p, ct.c_float, ct.c_int)
    print("CALL FUNCTION SubtractBgLoopMedianCutoff")
    averBG = pfun(_np_ptr(data),pix_nn, _np_ptr(pix_r), hitfinderMinSNR, ADCthresh, bstpReg, _np_ptr(radialcurve),cutoff, curveSmooth)
    return averBG



def pSubtractBgLoop(data ,pix_nn, pix_r, hitfinderMinSNR, ADCthresh, bstpReg, radialcurve): 
    #void SubtractBgLoop(float* data, long pix_nn, int *pix_r, float hitfinderMinSNR, float ADCthresh, float bstpReg, float* radialcurve)
    data = np.array(data, dtype=np.float32)
    pix_r = np.array(pix_r, dtype=np.int32)
    len_radialcurve = max(pix_r)
    # print(len_radialcurve)
    radialcurve = np.array([0]*len_radialcurve, dtype=np.float32)
    lib = ct.CDLL( '/home/galchenm/bin/main_prog/SubLocalBG.so' )
    pfun = lib.SubtractBgLoop
    #pfun.restype = ct.c_void
    pfun.restype = None
    pfun.argtypes = (ct.c_void_p, ct.c_long, ct.c_void_p, ct.c_float, ct.c_float, ct.c_float, ct.c_void_p)
    pfun(_np_ptr(data),pix_nn, _np_ptr(pix_r), hitfinderMinSNR, ADCthresh, bstpReg, _np_ptr(radialcurve))
    return radialcurve

#bool BuildRadialArray(size_t numEl, float *cox, float* coy, float istep, int* pix_r, int* maxRad, float* pixelsR)
def pBuildRadiulArray(NxNy, det_x, det_y, istep, pix_r, maxRad, pixelsR):
    det_x = np.array(det_x, dtype=np.float32)
    det_y = np.array(det_y, dtype=np.float32)
    
    # pix_r = np.zeros(NxNy,dtype=np.float32)
    # pixelsR = np.zeros(NxNy,dtype=np.float32)
    pix_r = np.array([0] * NxNy, dtype=np.int32)
    pixelsR = np.array([0] * NxNy, dtype=np.float32)
    # maxRad = 0
    maxRad = ct.c_int32()
    # pix_r = ct.c_float()
    # pixelsR = ct.c_float()
    
    lib = ct.CDLL( '/home/galchenm/bin/main_prog/SubLocalBG.so')
    pfun = lib.BuildRadialArray
    pfun.restype = ct.c_bool
    
    # pfun.argtypes = (ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_void_p, ct.c_void_p, ct.c_void_p)
    # flag_BRA = pfun(NxNy, _np_ptr(det_x), _np_ptr(det_y), istep, _np_ptr(pix_r), _np_ptr(maxRad), _np_ptr(pixelsR))
    pfun.argtypes = (ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_void_p, ct.POINTER(ct.c_int), ct.c_void_p)
    
    flag_BRA = pfun(NxNy, _np_ptr(det_x), _np_ptr(det_y), istep, _np_ptr(pix_r), ct.byref(maxRad), _np_ptr(pixelsR))
    
    return flag_BRA, pix_r, maxRad.value, pixelsR


def PolarisationFactorDet(x, y, z, degree = 0.99):
    
    pol = np.zeros_like(x)
    pdist2i = np.zeros_like(x)
    
    x_min_ind = np.where(np.fabs(x)<MinVal)[0]
    y_min_ind = np.where(np.fabs(y)<MinVal)[0]

    indices = range(0,len(x))
    min_ind = list((set(np.array(x_min_ind)) & set(np.array(y_min_ind))))

    not_min_ind = list(set(indices) - set(min_ind))
    
    pol[min_ind] = 1.0
    
    pdist2i[not_min_ind] = 1/np.hypot(x[not_min_ind],y[not_min_ind],z[not_min_ind])
    pol[not_min_ind] = 1 - ((y[not_min_ind] ** 2)*(1.-degree) + (x[not_min_ind] ** 2)*degree)*pdist2i[not_min_ind]

    pol_min = np.where(np.fabs(pol) < MinVal )
    pol[pol_min] = 1.0
        
    return pol

def apply_mask(Int, mask):
    print("Applying mask\n")
    mask = mask.flatten()
    intensity = np.zeros_like(Int)
    print('len(Int) in mask = ',len(Int))
    print('len(mask) in mask = ',len(mask))
    print('len(intensity) before = ', len(intensity))
    indexes_good = np.where(mask != MASK_BAD)[0]
    #indexes_bad = np.where(mask == MASK_BAD)[0]
    intensity[indexes_good] += Int[indexes_good]
    print('len(intensity) after = ', len(intensity))
    return intensity

def substract_profile(x_array, y_array, dist, Int, mask, mood, pol_bool, cutoff = 0.5 , curveSmooth = 1):
    print("--------Initialize substract process--------\n")
    len_array = len(x_array)
    z_array = np.ones_like(x_array) * dist
    print('len Int before mask = ', len(Int))
    Int = apply_mask(Int, mask)
    print('len Int after mask = ', len(Int))
    if pol_bool:
        print("Correct to polarisation factor\n")
        # calculate polarisation factor
        pol = PolarisationFactorDet(x.flatten(), y.flatten(), z, degree = 0.99)
        # Take into account polarisation
        Int *= pol

    # bool SubLocalBG(float* inpAr, int stx, int enx, int sty, int eny, int fNumX, int radX, int radY, float badVal, float* smoothedAr)
    #flag, Int = pSubLocalBG(Int, 0, 2463, 0, 2527, 2463, 3, 3, -0.1, None)
    #print("pSubLocalBG result: ", flag, Int)
    # bool BuildRadialArray(size_t numEl, float *cox, float* coy, float istep, int* pix_r, int* maxRad, float* pixelsR)
    flag_BRA, pix_r, maxRad, pixelsR = pBuildRadiulArray(len_array, x_array, y_array, 0.1, None, 0, None)
    #print("pBuildRadiulArray result: ", flag_BRA, pix_r, maxRad, pixelsR)
    print('len(Int) after pBuildRadiulArray = ', len(Int))
    #print("Mood: ", mood)

    if mood is 'median':
        print('len_array = ', len_array)
        r = np.hypot(x_array,y_array)
        r = r.astype(np.int)
        r = r.flatten()
        #SubtractBgLoopMedianCutoff(float* data, long pix_nn, int *pix_r, float hitfinderMinSNR, float ADCthresh, float bstpReg, float* radialcurve, float cutoff, int curveSmooth)
        # radialcurve = pSubtractBgLoopMedianCutoff(Int, len(Int), pix_r, 4.0, 0.0, 0.01, None, cutoff, curveSmooth)
        #radialcurve = 0.0
        #print("pSubtractBgLoopMedianCutoff result: ", radialcurve)
        print('len(Int) = ', len(Int))
        print('len(r) =', len(r))
        
        set_r = list(set(r))
        set_r.sort()
    
        max_r = max(set_r)

        median_r = np.array([0.]*len(r))
        m_curve = np.array([0.]*len(set_r))


        for j in set_r:
            result = np.where(r == j)[0]
            med_v = np.median(Int[result])
            median_r[result] += med_v
            m_curve[set_r.index(j)] += med_v


        '''
        b = len(set(r))
        radialcurve = stats.binned_statistic(r, Int, statistic = lambda q: np.nanmedian(q).astype(np.int), bins = b)[0]
        '''
        radialcurve = m_curve
        #med_background = median_r.reshape(data.shape)
        # Substract I = I - <I>
        #radialcurve = Int - median_r
        print("Own median result: ", radialcurve)

    else:
        # SubtractBgLoop(float* data, long pix_nn, int *pix_r, float hitfinderMinSNR, float ADCthresh, float bstpReg, float* radialcurve)
        radialcurve = pSubtractBgLoop(Int, len(Int), pix_r, 4.0, 0.0, 0.01, None)
        print("pSubtractBgLoop result: ", radialcurve)
    
    with open('browsers_median_low.txt', 'a+') as f:
        f.write(', '.join(map(str, radialcurve))+'\n')
            
   
    #return 0.0
    return radialcurve

def func_dist(data_name, geom_fnam):
    # get dist from geom file 
    pol_bool = False
    preamb = gf.read_geometry_file(geom_fnam, return_preamble = True)[8]
    dist_m = preamb['coffset']
    res = preamb['res']
    clen = preamb['clen']
    print('clen =', clen)
    check = data_name + clen
    print('check = ', check)
    myCmd = os.popen('h5ls ' + check).read()
    print('myCmd = ', myCmd)
    if "NOT" in myCmd:
        print('Error: no clen from .h5 file')
        dist = 0.
    else:
        pol_bool = True
        f = h5py.File(data_name, 'r')
        clen_v = f[clen].value * (1e-3)
        f.close()
        dist_m += clen_v
        dist = dist_m*res

    return dist, pol_bool

if __name__ == "__main__":
    data_file_name = sys.argv[1]
    geom_file = sys.argv[2]
    #mask = sys.argv[3]
    #mood = sys.argv[4] #median or radial
    #cutOff = sys.argv[5]
    #smooth = sys.argv[6]
    mood = 'median'
    pixm = cg.pixel_maps_from_geometry_file(geom_file)
    x_array = pixm.x
    y_array = pixm.y
    x_array = x_array.flatten()
    y_array = y_array.flatten()
    len_array = len(x_array)
    f = h5.File(data_file_name,'r') # h5.File(sys.argv[1],'r')
    data = f['data/data']
    data = np.reshape(data, len_array)
    #print("data in main: ", len(data))
    index = data >= 0
    mask = np.zeros(len_array, dtype=np.int32)
    mask[index] += MASK_GOOD
    #Int = cg.apply_geometry_from_file(data[:],geom_file)
    Int = data[:]
    print("Int in main: ", len(Int.flatten()))
    dist, pol_bool = func_dist(data_file_name, geom_file)
    print('dist = ', dist)
    # substract_profile(x_array, y_array, dist, Int, mask, mood, pol_bool, cutoff = 0.5 , curveSmooth = 1)
    r = substract_profile(x_array, y_array, dist, Int, mask, mood, pol_bool, 0.5,  1)