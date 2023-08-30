#!/usr/bin/env python
# coding: utf8

import os
import sys
import numpy as np
import math
import h5py as h5
from os.path import basename, splitext

from multiprocessing import Pool, Lock, RawArray
import ctypes as ct

import cfel_geom as cg

import numpy.linalg as ln

from struct import *
import re

MASK_GOOD = 1
MASK_BAD = 0

os.nice(0)

def _np_ptr(np_array):
    return ct.c_void_p(np_array.ctypes.data)

class PeakFinderStructure(ct.Structure):
    '''
    typedef struct {
    public:
        long        nPeaks;
        long        nHot;
        float       peakResolution;         // Radius of 80% of peaks
        float       peakResolutionA;        // Radius of 80% of peaks
        float       peakDensity;            // Density of peaks within this 80% figure
        float       peakNpix;               // Number of pixels in peaks
        float       peakTotal;              // Total integrated intensity in peaks
        int     memoryAllocated;
        long        nPeaks_max;

        float       *peak_maxintensity;     // Maximum intensity in peak
        float       *peak_totalintensity;   // Integrated intensity in peak
        float       *peak_sigma;            // Signal-to-noise ratio of peak
        float       *peak_snr;              // Signal-to-noise ratio of peak
        float       *peak_npix;             // Number of pixels in peak
        float       *peak_com_x;            // peak center of mass x (in raw layout)
        float       *peak_com_y;            // peak center of mass y (in raw layout)
        long        *peak_com_index;        // closest pixel corresponding to peak
        float       *peak_com_x_assembled;  // peak center of mass x (in assembled layout)
        float       *peak_com_y_assembled;  // peak center of mass y (in assembled layout)
        float       *peak_com_r_assembled;  // peak center of mass r (in assembled layout)
        float       *peak_com_q;            // Scattering vector of this peak
        float       *peak_com_res;          // REsolution of this peak
    } tPeakList;
    '''
    _fields_=[('nPeaks',ct.c_long), ('nHot',ct.c_long), ('peakResolution',ct.c_float), ('peakResolutionA',ct.c_float), ('peakDensity',ct.c_float), ('peakNpix',ct.c_float), 
                ('peakTotal',ct.c_float), ('memoryAllocated',ct.c_int), ('nPeaks_max',ct.c_long), ('peak_maxintensity',ct.POINTER(ct.c_float)), ('peak_totalintensity',ct.POINTER(ct.c_float)), 
                ('peak_sigma',ct.POINTER(ct.c_float)), ('peak_snr',ct.POINTER(ct.c_float)), ('peak_npix',ct.POINTER(ct.c_float)), ('peak_com_x',ct.POINTER(ct.c_float)), ('peak_com_y',ct.POINTER(ct.c_float)), ('peak_com_index',ct.POINTER(ct.c_long)), 
                ('peak_com_x_assembled',ct.POINTER(ct.c_float)), ('peak_com_y_assembled',ct.POINTER(ct.c_float)), ('peak_com_r_assembled',ct.POINTER(ct.c_float)), ('peak_com_q',ct.POINTER(ct.c_float)), ('peak_com_res',ct.POINTER(ct.c_float))]
    # pass
def PeakFinder8py(peaklist, data, mask, pix_r,
                    asic_nx, asic_ny, nasics_x, nasics_y,
                    ADCthresh, hitfinderMinSNR,
                    hitfinderMinPixCount, hitfinderMaxPixCount,
                    hitfinderLocalBGRadius, outliersMask):

    req = PeakFinderStructure()
    lib = ct.CDLL('peakfinder8.so')
    pfun = lib.peakfinder8
    pfun.restype = ct.c_int
    data = np.array(data, dtype=np.float32)
    pix_r = np.array(pix_r,dtype=np.float32)
    mask = np.array(mask, dtype=np.int8)
    len_outliersMask = len(data)
    outliersMask_buf = np.zeros(len_outliersMask, dtype=np.int8)

    pfun.argtypes = (ct.POINTER(PeakFinderStructure),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_long,ct.c_long,ct.c_long,ct.c_long,ct.c_float,ct.c_float,ct.c_long,ct.c_long,ct.c_long,ct.c_void_p)
    int_flag = pfun(ct.byref(req),_np_ptr(data),_np_ptr(mask),_np_ptr(pix_r),asic_nx, asic_ny, nasics_x, nasics_y,
                    ADCthresh, hitfinderMinSNR,
                    hitfinderMinPixCount, hitfinderMaxPixCount,
                    hitfinderLocalBGRadius, _np_ptr(outliersMask_buf))
    if outliersMask is not None:
        outliersMask[:] = outliersMask_buf.copy()
    return int_flag, outliersMask_buf


def background(inpAr, stx, enx, sty, eny, fNumX, radX, radY, badVal, smoothedAr):
    inpAr = np.array(inpAr, dtype=np.float32)
    lib = ct.CDLL( 'SubLocalBG.so' )
    pfun = lib.SubLocalBG 
    pfun.restype = ct.c_int
    pfun.argtypes = (ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_void_p)
    flag = pfun(_np_ptr(inpAr), stx, enx, sty, eny, fNumX, radX, radY, badVal, smoothedAr)
    return flag, inpAr

def MaskRingsSimplepy(data, mask, pix_r, numPo, badVal, ringDiff, smF):
    data = np.array(data, dtype=np.float32)
    pix_r = np.array(pix_r,dtype=np.int32)
    mask = np.array(mask, dtype=np.int8)
    lib = ct.CDLL( 'SubLocalBG.so' )
    pfun = lib.MaskRingsSimple
    pfun.restype = ct.c_int
    pfun.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_float, ct.c_float, ct.c_int)
    pfun(_np_ptr(data),_np_ptr(mask),_np_ptr(pix_r),numPo, badVal, ringDiff, smF)
    np.save('Int_after',data)
    return mask 
    
def pSubtractBgLoop(data ,pix_nn, pix_r, hitfinderMinSNR, ADCthresh, bstpReg, radialcurve): 
    data = np.array(data, dtype=np.float32)
    pix_r = np.array(pix_r, dtype=np.int32)
    len_radialcurve = max(pix_r)
    radialcurve = np.array([0]*len_radialcurve, dtype=np.float32)
    lib = ct.CDLL( 'SubLocalBG.so' )
    pfun = lib.SubtractBgLoop
    pfun.restype = None
    pfun.argtypes = (ct.c_void_p, ct.c_long, ct.c_void_p, ct.c_float, ct.c_float, ct.c_float, ct.c_void_p)
    pfun(_np_ptr(data),pix_nn, _np_ptr(pix_r), hitfinderMinSNR, ADCthresh, bstpReg, _np_ptr(radialcurve))
    return radialcurve

def pBuildRadiulArray(NxNy, det_x, det_y, istep, pix_r, maxRad, pixelsR):
    det_x = np.array(det_x, dtype=np.float32)
    det_y = np.array(det_y, dtype=np.float32)
    pix_r = np.array([0] * NxNy, dtype=np.int32)
    pixelsR = np.array([0] * NxNy, dtype=np.float32)
    maxRad = ct.c_int32()
    lib = ct.CDLL( 'SubLocalBG.so')
    pfun = lib.BuildRadialArray
    pfun.restype = ct.c_bool
    pfun.argtypes = (ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_void_p, ct.POINTER(ct.c_int), ct.c_void_p)
    flag_BRA = pfun(NxNy, _np_ptr(det_x), _np_ptr(det_y), istep, _np_ptr(pix_r), ct.byref(maxRad), _np_ptr(pixelsR))
    return flag_BRA, pix_r, maxRad.value, pixelsR


def hdf5_work(name_of_file,geom_file):
    f= h5.File(name_of_file,'r') 
    dataset = f['data/data'] # dataset = f[sys.argv[5]] # dataset [num] - the necessary image
    dataset_new = cg.apply_geometry_from_file(dataset[:],geom_file)
    plotting_hdf5(dataset_new)
    f.close()

def plot_Evald_sphere(qx,qy,qz,Int):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sp = ax.scatter(qx,qy,qz, s=20, c=Int)
    plt.colorbar(sp)
    plt.show()

def new_max_from_array(x, y, z):
    return (np.max(np.abs(np.array([x, y, z]))))

def NEW_sphere_Evald(qx, qy, qz, mask, Int, k):    
    global resolution
    global N_3D

    len_array = len(Int)

    cent = [(N_3D - 1) / 2.0] * 3

    qx = qx + cent[0]
    qy = qy + cent[1]
    qz = qz + cent[2]

    tmp_array = qz
    qz = qx
    qx = tmp_array

    indexes = np.where((qx < N_3D) &
                       (qy < N_3D) &
                       (qz < N_3D) &
                       (mask != MASK_BAD))[0]

    coords = np.array((qx, qy, qz), dtype=np.int32)[:, indexes]
    coords_intens = Int[indexes]
    intensity = np.histogramdd(coords.T,
                               bins=[np.arange(0, N_3D + 1, 1),
                                     np.arange(0, N_3D + 1, 1),
                                     np.arange(0, N_3D + 1, 1)],
                               normed=False,
                               weights=coords_intens)[0]

    click_count = np.histogramdd(coords.T,
                                 bins=[np.arange(0, N_3D + 1, 1),
                                       np.arange(0, N_3D + 1, 1),
                                       np.arange(0, N_3D + 1, 1)])[0]

    return intensity, click_count


def cell_crystallographic_to_cartesian(cell_param):  # cell_param.append([float(a),float(b),float(c),float(alfa),float(betta),float(gamma)])
    mat_id = []

    alfa = cell_param[1][0] * math.pi / 180.0
    betta = cell_param[1][1] * math.pi / 180.0
    gamma = cell_param[1][2] * math.pi / 180.0

    astar_x = cell_param[0][0]
    astar_y = 0.0
    astar_z = 0.0

    bstar_x = cell_param[0][1] * math.cos(gamma)
    bstar_y = cell_param[0][1] * math.sin(gamma)
    bstar_z = 0.0

    tmp = math.cos(alfa) * math.cos(alfa) + math.cos(betta) * math.cos(betta) + math.cos(gamma) * math.cos(
        gamma) - 2.0 * math.cos(alfa) * math.cos(betta) * math.cos(gamma)
    V = cell_param[0][0] * cell_param[0][1] * cell_param[0][2] * math.sqrt(1.0 - tmp)

    cosalphastar = math.cos(betta) * math.cos(gamma) - math.cos(alfa)
    cosalphastar /= math.sin(betta) * math.sin(gamma)

    cstar = (cell_param[0][0] * cell_param[0][1] * math.sin(gamma)) / V

    cstar_x = cell_param[0][2] * math.cos(betta)
    cstar_y = -cell_param[0][2] * math.sin(betta) * cosalphastar
    cstar_z = 1.0 / cstar

    mat_id.append([astar_x, astar_y, astar_z])
    mat_id.append([bstar_x, bstar_y, bstar_z])
    mat_id.append([cstar_x, cstar_y, cstar_z])

    mat_id = np.array(mat_id)

    return mat_id


def create_mat_cur(astar, bstar, cstar):
    mat_cur = []
    mat_cur.append(astar)
    mat_cur.append(bstar)
    mat_cur.append(cstar)
    mat_cur = np.array(mat_cur)
    mat_cur = mat_cur.T
    return mat_cur

def NEW_sphere_Evald_with_rotation(qx,qy,qz,
                                   mask, Int, k, mat_cur, cell_param):
    # global resolution
    global N_3D

    len_array = len(Int)
    cent = [(N_3D - 1) / 2.0] * 3

    mat_id = cell_crystallographic_to_cartesian(cell_param)

    transform_matrix = mat_cur.dot(mat_id)
    transform_matrix = ln.inv(transform_matrix)

    qx, qy, qz = np.dot(transform_matrix, np.array([qx, qy, qz]))

    qx = qx + cent[0]
    qy = qy + cent[1]
    qz = qz + cent[2]

    tmp_array = qz
    qz = qx
    qx = tmp_array

    indexes = np.where((qx < N_3D) &
                       (qy < N_3D) &
                       (qz < N_3D) &
                       (mask != MASK_BAD))[0]

    coords = np.array((qx, qy, qz), dtype=np.int32)[:, indexes]
    coords_intens = Int[indexes]
    intensity = np.histogramdd(coords.T,
                               bins=[np.arange(0, N_3D + 1, 1),
                                     np.arange(0, N_3D + 1, 1),
                                     np.arange(0, N_3D + 1, 1)],
                               normed=False,
                               weights=coords_intens)[0]

    click_count = np.histogramdd(coords.T,
                                 bins=[np.arange(0, N_3D + 1, 1),
                                       np.arange(0, N_3D + 1, 1),
                                       np.arange(0, N_3D + 1, 1)])[0]

    return intensity, click_count


def get_opt_patterns(lines):
    metadata = []

    name_of_file = None
    photon_energy_eV = None
    average_camera_length = None
    cell = None
    astar = None
    bstar = None
    cstar = None

    for line in lines:
        if 'Image filename:' in line:
            name_of_file = line.split()[2]

        elif 'photon_energy_eV =' in line:
            photon_energy_eV = float(line.split('=')[1])

        elif 'average_camera_length =' in line:
            average_camera_length = float(line.split(' ')[2])

        elif 'Cell parameters' in line:
            pref3, pref4, a, b, c, size1, alfa, betta, gamma, size_deg = line.split(' ')
            cell = np.array([[float(a) * 1e-9,
                              float(b) * 1e-9,
                              float(c) * 1e-9],
                             [float(alfa),
                              float(betta),
                              float(gamma)]],
                            dtype=float)


        elif 'astar =' in line:
            pref5, sig1, astar1, astar2, astar3, s1 = line.split(' ')
            astar = np.array([float(astar1),
                              float(astar2),
                              float(astar3)],
                             dtype=float) * 1e9

        elif 'bstar = ' in line:
            pref5, sig1, bstar1, bstar2, bstar3, s2 = line.split(' ')
            bstar = np.array([float(bstar1),
                              float(bstar2),
                              float(bstar3)],
                             dtype=float) * 1e9


        elif 'cstar =' in line:
            pref5, sig1, cstar1, cstar2, cstar3, s2 = line.split(' ')
            cstar = np.array([float(cstar1),
                              float(cstar2),
                              float(cstar3)],
                             dtype=float) * 1e9

        if (name_of_file is not None) and (photon_energy_eV is not None) and (average_camera_length is not None) and (
            cell is not None) and (astar is not None) and (bstar is not None) and (cstar is not None):
            metadata.append([name_of_file, photon_energy_eV, average_camera_length, cell, astar, bstar, cstar])
            name_of_file = None
            photon_energy_eV = None
            average_camera_length = None
            cell = None
            astar = None
            bstar = None
            cstar = None

    return metadata


def processing(i):
    global result_lock

    global I_xyz_arr
    global count_dot_arr
    global r
    global coords
    len_array = len(x_array)

    name_of_file = i[0]
    f = h5.File(name_of_file, 'r')  # h5.File(sys.argv[1],'r')
    Int = f['data/data']
    Int = np.reshape(Int, len_array)

    index = Int >= 0

    mask = np.zeros(len_array, dtype=np.int32)
    mask[index] += MASK_GOOD
    

    np.save('mask_before',mask)

    np.save('Int_before',Int)
    # flag,Int = background(Int, 0, 2463, 0, 2527, 2463, 3, 3, -0.1, None)
    #np.save('data_intensity_after',Int)
    
    flag_BRA, pix_r, maxRad, pixelsR = pBuildRadiulArray(len_array, x_array, y_array, 1/172e-6, None, 0, None) #pBuildRadiulArray(len_array, x_array, y_array, 1.0)
    # print(flag_BRA, pix_r, maxRad, pixelsR)

    # radialcurve = pSubtractBgLoop(Int, len(Int), pix_r, 4.0, 0.0, 0.01, None)


    # int_PeakFinder_flag, outliersMask = PeakFinder8py(None, Int, mask, pixelsR,
    #                 2463, 2527, 1, 1,
    #                 10, 4,
    #                 2, 1,
    #                 10, None)
    
    #(float* data, char *mask, int *pix_r, int numPo, float badVal, float ringDiff, int smF)
    # np.save('mask_before',mask)
    mask = MaskRingsSimplepy(Int, mask, pix_r, len(x_array), -10000, 0.1, 0)

    np.save('mask_after',mask)
    
    # np.save('outliersMask',outliersMask)

    k = i[1] * (1e10) / 12.4
    average_camera_length = i[2]
    cell_param = i[3]

    mat_cur = create_mat_cur(i[4], i[5], i[6])
    
    twotheta = np.arctan2(r, coords[2])
    az = np.arctan2(coords[1], coords[0])

    qx = np.sin(twotheta) * np.cos(az) * k
    qy = np.sin(twotheta) * np.sin(az) * k
    qz = (np.cos(twotheta) - 1.0) * k

    qx = np.array(qx)
    qy = np.array(qy)
    qz = np.array(qz)

    scale = N_3D / (2 * new_max_from_array(qx, qy, qz))

    qx = qx * scale
    qy = qy * scale
    qz = qz * scale


    intensity, click_count = NEW_sphere_Evald_with_rotation(qx,qy,qz,
                                   mask, Int, k, mat_cur, cell_param)
    with result_lock:
        I_xyz_arr += intensity
        count_dot_arr += click_count.astype(count_dot_arr.dtype)



def init_worker(res_lck, I_xyz_buf, count_dot_buf,
                r_arr,coords_arr):
    global result_lock

    global I_xyz_arr
    global count_dot_arr

    global r
    global coords


    I_xyz_arr = np.frombuffer(I_xyz_buf, dtype=np.double)
    I_xyz_arr = np.reshape(I_xyz_arr,(N_3D,N_3D,N_3D))
    count_dot_arr = np.frombuffer(count_dot_buf, dtype=np.double)
    count_dot_arr = np.reshape(count_dot_arr,(N_3D,N_3D,N_3D))
    result_lock = res_lck
    r = r_arr
    coords = coords_arr



if __name__ == "__main__":
    geom_file = 'pilatus6M_200mm_man1.geom'
    event = None
    metadata = []
    cell_param = []
    resolution = 5814.0  # sys.argv[6] from geom file
    dist = 0.2
    N_3D = 501  # num of points

    output_array = []

    pixm = cg.pixel_maps_from_geometry_file(geom_file)
    x_array = pixm.x.flatten()
    y_array = pixm.y.flatten()
    z_array = np.ones_like(x_array) * dist
    len_array = len(x_array)

    x_array = x_array * (1 / resolution)
    y_array = y_array * (1 / resolution)

    coords = np.array((x_array, y_array, z_array))

    r = np.sqrt(np.square(coords[0]) + np.square(coords[1]))

    with open('lys_14a_02_001_tr.stream','r') as stream:
        lines = stream.readlines()
        output_array = get_opt_patterns(lines)
    
    I_xyz_buf = RawArray(ct.c_double, N_3D * N_3D * N_3D)
    count_dot_buf = RawArray(ct.c_double, N_3D * N_3D * N_3D)
    result_lock = Lock()
    pool = Pool(31,
                initializer=init_worker,
                initargs=(result_lock, I_xyz_buf, count_dot_buf,
                          r,coords))

    pool.map(processing, output_array)
    I_xyz = np.frombuffer(I_xyz_buf, dtype=np.double)
    count_dot = np.frombuffer(count_dot_buf, dtype=np.double)
    
    index = (count_dot!= 0)
    I_xyz[index] =I_xyz[index] / count_dot[index]
    np.save('Evald_sphere_processing.npy',I_xyz)
    