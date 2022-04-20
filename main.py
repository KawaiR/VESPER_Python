import argparse
import concurrent.futures
import copy
import csv
import math

import mrcfile
import multiprocessing
import numba
import numpy as np
import os
import pyfftw
import scipy.fft
from VESPER_1_prob_new import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')

    orig = subparser.add_parser('orig')
    prob = subparser.add_parser('prob')

    # original VESPER menu
    orig.add_argument('-a', type=str, required=True, help='MAP1.mrc (large)')
    orig.add_argument('-b', type=str, required=True, help='MAP2.mrc (small)')
    orig.add_argument('-t', type=float, help='Threshold of density map1')
    orig.add_argument('-T', type=float, help='Threshold of density map2')
    orig.add_argument('-g', type=float, default=16.0,
                      help='Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]')
    orig.add_argument('-s', type=float, default=7.0, help='Sampling voxel spacing def=7.0')
    orig.add_argument('-A', type=float, default=30.0, help='Sampling angle spacing def=30.0')
    orig.add_argument('-N', type=int, default=10, help='Refine Top [int] models def=10')
    orig.add_argument('-S', type=bool, default=False, help='Show topN models in PDB format def=false')
    orig.add_argument('-M', type=str, default='V', help='V: vector product mode (default)\n' +
                                                        'O: overlap mode\n' +
                                                        'C: Cross Correlation Coefficient Mode\n' +
                                                        'P: Pearson Correlation Coefficient Mode\n' +
                                                        'L: Laplacian Filtering Mode')
    orig.add_argument('-E', type=bool, default=False, help='Evaluation mode of the current position def=false')

    # secondary structure matching menu
    prob.add_argument('-a', type=str, required=True, help='MAP1.mrc (large)')
    prob.add_argument('-npa', type=str, required=True, help='numpy array for Predictions for map 1')
    prob.add_argument('-b', type=str, required=True, help='MAP2.mrc (small)')
    prob.add_argument('-npb', type=str, required=True, help='numpy array for Predictions for map 2')
    prob.add_argument('-alpha', type=float, default=0.5, required=False, help='The weighting parameter for alpha '
                                                                              'mixing def=0.0')
    prob.add_argument('-t', type=float, help='Threshold of density map1')
    prob.add_argument('-T', type=float, help='Threshold of density map2')
    prob.add_argument('-g', type=float, default=16.0,
                      help='Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]')
    prob.add_argument('-s', type=float, default=7.0, help='Sampling voxel spacing def=7.0')
    prob.add_argument('-A', type=float, default=30.0, help='Sampling angle spacing def=30.0')
    prob.add_argument('-N', type=int, default=10, help='Refine Top [int] models def=10')
    prob.add_argument('-S', type=bool, default=False, help='Show topN models in PDB format def=false')
    prob.add_argument('-M', type=str, default='V', help='V: vector product mode (default)\n' +
                                                        'O: overlap mode\n' +
                                                        'C: Cross Correlation Coefficient Mode\n' +
                                                        'P: Pearson Correlation Coefficient Mode\n' +
                                                        'L: Laplacian Filtering Mode')
    prob.add_argument('-E', type=bool, default=False, help='Evaluation mode of the current position def=false')
    prob.add_argument('-P', type=int, default=4, help='Number of processors to use def=4')
    prob.add_argument('-vav', type=float, help='Pre-computed average for density map')
    prob.add_argument('-vstd', type=float, help='Pre-computed standard deviation for density map')
    prob.add_argument('-pav', type=float, help='Pre-computed average for probability map')
    prob.add_argument('-pstd', type=float, help='Pre-computed standard deviation for probability map')

    args = parser.parse_args()

    if args.command == 'orig':
        # mrc file paths
        objA = args.a
        objB = args.b

        # all optional positional arguments
        # check for none
        threshold1 = args.t
        threshold2 = args.T

        if threshold1 is None:
            threshold1 = 0.00

        if threshold2 is None:
            threshold2 = 0.00

        # with default
        bandwidth = args.g
        voxel_spacing = args.s
        angle_spacing = args.A
        topN = args.N

        # necessary?
        showPdb = args.S

        modeVal = args.M
        evalMode = args.E

        print(objA, objB, threshold1, threshold2, bandwidth, voxel_spacing, angle_spacing, topN, showPdb, modeVal,
              evalMode)

        # construct mrc objects
        mrc1 = mrc_obj(objA)
        mrc2 = mrc_obj(objB)

        # set voxel size
        mrc1, mrc_N1 = mrc_set_vox_size(mrc1, threshold1, voxel_spacing)
        mrc2, mrc_N2 = mrc_set_vox_size(mrc2, threshold2, voxel_spacing)

        if mrc_N1.xdim > mrc_N2.xdim:
            dim = mrc_N2.xdim = mrc_N2.ydim = mrc_N2.zdim = mrc_N1.xdim

            mrc_N2.orig[0] = mrc_N2.cent[0] - 0.5 * voxel_spacing * mrc_N2.xdim
            mrc_N2.orig[1] = mrc_N2.cent[1] - 0.5 * voxel_spacing * mrc_N2.xdim
            mrc_N2.orig[2] = mrc_N2.cent[2] - 0.5 * voxel_spacing * mrc_N2.xdim

        else:
            dim = mrc_N1.xdim = mrc_N1.ydim = mrc_N1.zdim = mrc_N2.xdim

            mrc_N1.orig[0] = mrc_N1.cent[0] - 0.5 * voxel_spacing * mrc_N1.xdim
            mrc_N1.orig[1] = mrc_N1.cent[1] - 0.5 * voxel_spacing * mrc_N1.xdim
            mrc_N1.orig[2] = mrc_N1.cent[2] - 0.5 * voxel_spacing * mrc_N1.xdim

        mrc_N1.dens = np.zeros((dim ** 3, 1))
        mrc_N1.vec = np.zeros((dim, dim, dim, 3), dtype="float32")
        mrc_N1.data = np.zeros((dim, dim, dim))
        mrc_N2.dens = np.zeros((dim ** 3, 1))
        mrc_N2.vec = np.zeros((dim, dim, dim, 3), dtype="float32")
        mrc_N2.data = np.zeros((dim, dim, dim))

        # fastVEC
        mrc_N1 = fastVEC(mrc1, mrc_N1, bandwidth)
        mrc_N2 = fastVEC(mrc2, mrc_N2, bandwidth)

        # search map
        if modeVal == 'V':
            modeVal = "VecProduct"
        elif modeVal == 'O':
            modeVal = "Overlap"
        elif modeVal == 'C':
            modeVal = "CC"
        elif modeVal == 'P':
            modeVal = "PCC"
        elif modeVal == 'L':
            modeVal = "Laplacian"

        search_map_fft(mrc_N1, mrc_N2, TopN=topN, ang=angle_spacing, mode=modeVal, is_eval_mode=evalMode)


    elif args.command == 'prob':
        # mrc file paths
        objA = args.a
        objB = args.b

        # numpy arrays

        npA = args.npa
        npB = args.npb

        # probability map file paths
        probA = args.a
        probB = args.b

        # all optional positional arguments
        # check for none
        threshold1 = args.t
        threshold2 = args.T

        # weighting parameter
        alpha = args.alpha

        if threshold1 is None:
            threshold1 = 0.00

        if threshold2 is None:
            threshold2 = 0.00

        # with default
        bandwidth = args.g
        voxel_spacing = args.s
        angle_spacing = args.A
        topN = args.N

        #average and standard deviation
        vave = args.vav
        vstd = args.vstd
        pave = args.pav
        pstd = args.pstd

        if vave is None:
            vave = -10
        if vstd is None:
            vstd = -10
        if pave is None:
            pave = -10
        if pstd is None:
            pstd = -10

        # necessary?
        showPdb = args.S

        modeVal = args.M
        evalMode = args.E

        num_processors = args.P

        print(objA, objB, probA, probB, threshold1, threshold2, bandwidth, voxel_spacing, angle_spacing, topN, showPdb,
              modeVal, evalMode, vave, vstd, pave, pstd)

        prob_maps = np.load(npA)
        prob_maps_chain = np.load(npB)

        mrc1 = mrc_obj(objA)
        mrc2 = mrc_obj(objB)

        # probability MRCs

        mrc1_p1 = mrc_obj(objA)
        mrc1_p2 = mrc_obj(objA)
        mrc1_p3 = mrc_obj(objA)
        mrc1_p4 = mrc_obj(objA)

        mrc2_p1 = mrc_obj(objB)
        mrc2_p2 = mrc_obj(objB)
        mrc2_p3 = mrc_obj(objB)
        mrc2_p4 = mrc_obj(objB)

        mrc2_p1.data = np.swapaxes(prob_maps_chain[:, :, :, 0], 0, 2)
        mrc2_p2.data = np.swapaxes(prob_maps_chain[:, :, :, 1], 0, 2)
        mrc2_p3.data = np.swapaxes(prob_maps_chain[:, :, :, 2], 0, 2)
        mrc2_p4.data = np.swapaxes(prob_maps_chain[:, :, :, 3], 0, 2)

        mrc1_p1.data = np.swapaxes(prob_maps[:, :, :, 0], 0, 2)
        mrc1_p2.data = np.swapaxes(prob_maps[:, :, :, 1], 0, 2)
        mrc1_p3.data = np.swapaxes(prob_maps[:, :, :, 2], 0, 2)
        mrc1_p4.data = np.swapaxes(prob_maps[:, :, :, 3], 0, 2)

        print("\n###Generating Params for Resampling Density Map 1###")
        mrc1, mrc_N1 = mrc_set_vox_size(mrc1, threshold1, voxel_spacing)

        print("\n###Generating Params for Resampling Density Map 2###")
        mrc2, mrc_N2 = mrc_set_vox_size(mrc2, threshold2, voxel_spacing)

        print("\n###Generating Params for Resampling Probability Map 1###")

        mrc1_p1, mrc_N1_p1 = mrc_set_vox_size(mrc1_p1, th=0.0, voxel_size=voxel_spacing)
        mrc1_p2, mrc_N1_p2 = mrc_set_vox_size(mrc1_p2, th=0.0, voxel_size=voxel_spacing)
        mrc1_p3, mrc_N1_p3 = mrc_set_vox_size(mrc1_p3, th=0.0, voxel_size=voxel_spacing)
        mrc1_p4, mrc_N1_p4 = mrc_set_vox_size(mrc1_p4, th=0.0, voxel_size=voxel_spacing)

        print("\n###Generating Params for Resampling Probability Map 2###")

        mrc2_p1, mrc_N2_p1 = mrc_set_vox_size(mrc2_p1, th=0.0, voxel_size=voxel_spacing)
        mrc2_p2, mrc_N2_p2 = mrc_set_vox_size(mrc2_p2, th=0.0, voxel_size=voxel_spacing)
        mrc2_p3, mrc_N2_p3 = mrc_set_vox_size(mrc2_p3, th=0.0, voxel_size=voxel_spacing)
        mrc2_p4, mrc_N2_p4 = mrc_set_vox_size(mrc2_p4, th=0.0, voxel_size=voxel_spacing)

        mrc_list = [mrc_N1, mrc_N2, mrc_N1_p1, mrc_N1_p2, mrc_N1_p3, mrc_N1_p4, mrc_N2_p1, mrc_N2_p2, mrc_N2_p3,
                    mrc_N2_p4]
        max_dim = np.max((mrc_N1.xdim, mrc_N2.xdim, mrc_N2_p1.xdim, mrc_N2_p2.xdim, mrc_N2_p3.xdim, mrc_N2_p4.xdim,
                          mrc_N1_p1.xdim, mrc_N1_p2.xdim, mrc_N1_p3.xdim, mrc_N1_p4.xdim))

        # Unify dimensions in all maps

        for mrc in mrc_list:
            if mrc.xdim != max_dim:
                mrc.xdim = mrc.ydim = mrc.zdim = max_dim
                mrc.orig[0] = mrc.cent[0] - 0.5 * voxel_spacing * mrc.xdim
                mrc.orig[1] = mrc.cent[1] - 0.5 * voxel_spacing * mrc.xdim
                mrc.orig[2] = mrc.cent[2] - 0.5 * voxel_spacing * mrc.xdim

        for mrc in mrc_list:
            mrc.dens = np.zeros((max_dim ** 3, 1), dtype="float32")
            mrc.data = np.zeros((max_dim, max_dim, max_dim), dtype="float32")
            mrc.vec = np.zeros((max_dim, max_dim, max_dim, 3), dtype="float32")

        print("\n###Processing MAP1 Resampling###")

        mrc_N1 = fastVEC(mrc1, mrc_N1, dreso=bandwidth)
        mrc_N1_p1 = fastVEC(mrc1_p1, mrc_N1_p1, dreso=bandwidth)
        mrc_N1_p2 = fastVEC(mrc1_p2, mrc_N1_p2, dreso=bandwidth)
        mrc_N1_p3 = fastVEC(mrc1_p3, mrc_N1_p3, dreso=bandwidth)
        mrc_N1_p4 = fastVEC(mrc1_p4, mrc_N1_p4, dreso=bandwidth)

        print("\n###Processing MAP2 Resampling###")
        mrc_N2 = fastVEC(mrc2, mrc_N2, dreso=bandwidth)
        mrc_N2_p1 = fastVEC(mrc2_p1, mrc_N2_p1, dreso=bandwidth)
        mrc_N2_p2 = fastVEC(mrc2_p2, mrc_N2_p2, dreso=bandwidth)
        mrc_N2_p3 = fastVEC(mrc2_p3, mrc_N2_p3, dreso=bandwidth)
        mrc_N2_p4 = fastVEC(mrc2_p4, mrc_N2_p4, dreso=bandwidth)

        search_map_fft_prob(mrc_N1_p1, mrc_N1_p2, mrc_N1_p3, mrc_N1_p4,
                            mrc_N1, mrc_N2,
                            mrc_N2_p1, mrc_N2_p2, mrc_N2_p3, mrc_N2_p4,
                            ang=angle_spacing, alpha=alpha, TopN=topN, num_proc=num_processors, vave=vave, vstd=vstd,
                            pave = pave, pstd = pstd)
