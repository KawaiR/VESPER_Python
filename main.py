import concurrent.futures
import copy
import math
import multiprocessing
import os
import csv

import argparse
from typing import Text
import mkdir
import mrcfile
import numba
import numpy as np
import pyfftw
import scipy.fft
from numba.typed import List
from scipy.ndimage import convolve
from scipy.spatial.transform import Rotation as R
#from tqdm.notebook import tqdm

from VESPER_1 import *


if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument('--a', type=str, required=True, help='MAP1.mrc (large)')
    params.add_argument('--b', type=str, required=True, help='MAP2.mrc (small)')
    params.add_argument('--t', type=float, help='Threshold of density map1')
    params.add_argument('--T', type=float, help='Threshold of density map2')
    params.add_argument('--g', type=float, default=16.0, help='Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]')
    params.add_argument('--s', type=float, default=7.0, help='Sampling voxel spacing def=7.0')
    params.add_argument('--A', type=float, default=30.0, help='Sampling angle spacing def=30.0')
    params.add_argument('--N', type=int, default=10, help='Refine Top [int] models def=10')
    params.add_argument('--S', type=bool, default=False, help='Show topN models in PDB format def=false')
    params.add_argument('--M', type=str, default='V', help='V: vector product mode (default)\nO: overlap mode\nC: Cross Correlation Coefficient Mode\nP: Pearson Correlation Coefficient Mode\nL: Laplacian Filtering Mode')
    params.add_argument('--E', type=bool, default=False, help='Evaluation mode of the current position def=false')
    args = params.parse_args()
    
    #mrc file paths
    objA = args.a
    objB = args.b

    #all optional positional arguments
    #check for none
    threshold1 = args.t
    threshold2 = args.T

    if threshold1 is None:
        threshold1 = 0.01
    
    if threshold2 is None:
        threshold2 = 0.01
    
    #with default
    bandwidth = args.g
    voxel_spacing = args.s
    angle_spacing = args.A
    topN = args.N

    #necessary?
    showPdb = args.S
    
    modeVal = args.M
    evalMode = args.E

    print(objA, objB, threshold1, threshold2, bandwidth, voxel_spacing, angle_spacing, topN, showPdb, modeVal, evalMode)
    
    #construct mrc objects
    mrc1 = mrc_obj(objA)
    mrc2 = mrc_obj(objB)

    #set voxel size
    mrc1, mrc_N1 = mrc_set_vox_size(mrc1, threshold1, voxel_spacing)
    mrc2, mrc_N2 = mrc_set_vox_size(mrc2, threshold2, voxel_spacing)

    if mrc_N1.xdim > mrc_N2.xdim:
        dim = mrc_N2.xdim = mrc_N2.ydim = mrc_N2.zdim = mrc_N1.xdim

        mrc_N2.orig["x"] = mrc_N2.cent[0] - 0.5 * voxel_spacing * mrc_N2.xdim
        mrc_N2.orig["y"] = mrc_N2.cent[1] - 0.5 * voxel_spacing * mrc_N2.xdim
        mrc_N2.orig["z"] = mrc_N2.cent[2] - 0.5 * voxel_spacing * mrc_N2.xdim

    else:
        dim = mrc_N1.xdim = mrc_N1.ydim = mrc_N1.zdim = mrc_N2.xdim

        mrc_N1.orig["x"] = mrc_N1.cent[0] - 0.5 * voxel_spacing * mrc_N1.xdim
        mrc_N1.orig["y"] = mrc_N1.cent[1] - 0.5 * voxel_spacing * mrc_N1.xdim
        mrc_N1.orig["z"] = mrc_N1.cent[2] - 0.5 * voxel_spacing * mrc_N1.xdim

    mrc_N1.dens = np.zeros((dim ** 3, 1))
    mrc_N1.vec = np.zeros((dim, dim, dim, 3), dtype="float32")
    mrc_N1.data = np.zeros((dim, dim, dim))
    mrc_N2.dens = np.zeros((dim ** 3, 1))
    mrc_N2.vec = np.zeros((dim, dim, dim, 3), dtype="float32")
    mrc_N2.data = np.zeros((dim, dim, dim))

    #fastVEC
    mrc_N1 = fastVEC(mrc1, mrc_N1, bandwidth)
    mrc_N2 = fastVEC(mrc2, mrc_N2, bandwidth)

    #search map
    if modeVal == 'V':
        modeVal="vecProduct"
    elif modeVal == 'O':
        modeVal="Overlap"
    elif modeVal == 'C':
        modeVal="CC"
    elif modeVal == 'P':
        modeVal="PCC"
    elif modeVal == 'L':
        modeVal="Laplacian"
    
    search_map_fft(mrc_N1, mrc_N2, TopN=topN, ang=angle_spacing, mode=modeVal, is_eval_mode=evalMode)
