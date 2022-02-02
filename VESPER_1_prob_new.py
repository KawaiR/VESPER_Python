# coding: utf-8

import concurrent.futures
import copy
import math
import multiprocessing
import os

import mrcfile
import numba
import numpy as np
import pyfftw
import scipy.fft
from numba.typed import List
from scipy.ndimage import convolve, correlate
from scipy.spatial.transform import Rotation as R
from tqdm.notebook import tqdm
#import cupy as cp

pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

class mrc_obj:
    def __init__(self, path):
        mrc = mrcfile.open(path)
        data = mrc.data
        header = mrc.header
        self.xdim = int(header.nx)
        self.ydim = int(header.ny)
        self.zdim = int(header.nz)
        self.xwidth = mrc.voxel_size.x
        self.ywidth = mrc.voxel_size.y
        self.zwidth = mrc.voxel_size.z
        self.cent = [
            self.xdim * 0.5,
            self.ydim * 0.5,
            self.zdim * 0.5,
        ]
        self.orig = {"x": header.origin.x, "y": header.origin.y, "z": header.origin.z}
        self.data = np.swapaxes(copy.deepcopy(data), 0, 2)
        self.dens = data.flatten()
        self.vec = np.zeros((self.xdim, self.ydim, self.zdim, 3), dtype="float32")
        self.dsum = None
        self.Nact = None
        self.ave = None
        self.std_norm_ave = None
        self.std = None

def mrc_set_vox_size(mrc, th=0.01, voxel_size=7.0):

    # set shape and size
    size = mrc.xdim * mrc.ydim * mrc.zdim
    shape = (mrc.xdim, mrc.ydim, mrc.zdim)

    # if th < 0 add th to all value
    if th < 0:
        mrc.dens = mrc.dens - th
        th = 0.0

    # Trim all the values less than threshold
    mrc.dens[mrc.dens < th] = 0.0
    mrc.data[mrc.data < th] = 0.0

    # calculate dmax distance for non-zero entries
    non_zero_index_list = np.array(np.nonzero(mrc.data)).T
    # non_zero_index_list[:, [2, 0]] = non_zero_index_list[:, [0, 2]]
    cent_arr = np.array(mrc.cent)
    d2_list = np.linalg.norm(non_zero_index_list - cent_arr, axis=1)
    dmax = max(d2_list)

    # dmax = math.sqrt(mrc.cent[0] ** 2 + mrc.cent[1] ** 2 + mrc.cent[2] ** 2)

    print("#dmax=" + str(dmax / mrc.xwidth))
    dmax = dmax * mrc.xwidth

    # set new center
    new_cent = [
        mrc.cent[0] * mrc.xwidth + mrc.orig["x"],
        mrc.cent[1] * mrc.xwidth + mrc.orig["y"],
        mrc.cent[2] * mrc.xwidth + mrc.orig["z"],
    ]

    tmp_size = 2 * dmax / voxel_size

    new_xdim = pyfftw.next_fast_len(int(tmp_size))

    # set new origins
    new_orig = {
        "x": new_cent[0] - 0.5 * new_xdim * voxel_size,
        "y": new_cent[1] - 0.5 * new_xdim * voxel_size,
        "z": new_cent[2] - 0.5 * new_xdim * voxel_size,
    }

    # create new mrc object
    mrc_set = copy.deepcopy(mrc)
    mrc_set.orig = new_orig
    mrc_set.xdim = new_xdim
    mrc_set.ydim = new_xdim
    mrc_set.zdim = new_xdim
    mrc_set.cent = new_cent
    mrc_set.xwidth = mrc_set.ywidth = mrc_set.zwidth = voxel_size

    print("Nvox= " + str(mrc_set.xdim) + ", " + str(mrc_set.ydim) + ", " + str(mrc_set.zdim))
    print("cent= " + str(new_cent[0]) + ", " + str(new_cent[1]) + ", " + str(new_cent[2]))
    print("ori= " + str(new_orig["x"]) + ", " + str(new_orig["y"]) + ", " + str(new_orig["z"]))

    return mrc, mrc_set


@numba.jit(nopython=True)
def calc(stp, endp, pos, mrc1_data, fsiv):
    dtotal = 0.0
    pos2 = np.zeros((3,))

    for xp in range(stp[0], endp[0]):
        rx = float(xp) - pos[0]
        rx = rx ** 2
        for yp in range(stp[1], endp[1]):
            ry = float(yp) - pos[1]
            ry = ry ** 2
            for zp in range(stp[2], endp[2]):
                rz = float(zp) - pos[2]
                rz = rz ** 2
                d2 = rx + ry + rz
                v = mrc1_data[xp][yp][zp] * math.exp(-1.5 * d2 * fsiv)
                dtotal += v
                pos2[0] += v * float(xp)
                pos2[1] += v * float(yp)
                pos2[2] += v * float(zp)
                
    return dtotal, pos2


def fastVEC(mrc1, mrc2, dreso=16.0):

    xydim = mrc1.xdim * mrc1.ydim
    Ndata = mrc2.xdim * mrc2.ydim * mrc2.zdim

    print(len(mrc2.dens))

    print("#Start VEC")
    gstep = mrc1.xwidth
    fs = (dreso / gstep) * 0.5
    fs = fs ** 2
    fsiv = 1.0 / fs
    fmaxd = (dreso / gstep) * 2.0
    print("#maxd= {fmaxd}".format(fmaxd=fmaxd))
    print("#fsiv= " + str(fsiv))

    dsum = 0.0
    Nact = 0

    list_d = []

    for x in tqdm(range(mrc2.xdim)):
        for y in range(mrc2.ydim):
            for z in range(mrc2.zdim):
                stp = [0] * 3
                endp = [0] * 3
                ind2 = 0
                ind = 0

                pos = [0.0] * 3
                pos2 = [0.0] * 3
                ori = [0.0] * 3

                tmpcd = [0.0] * 3

                v, dtotal, rd = 0.0, 0.0, 0.0

                pos[0] = (x * mrc2.xwidth + mrc2.orig["x"] - mrc1.orig["x"]) / mrc1.xwidth
                pos[1] = (y * mrc2.xwidth + mrc2.orig["y"] - mrc1.orig["y"]) / mrc1.xwidth
                pos[2] = (z * mrc2.xwidth + mrc2.orig["z"] - mrc1.orig["z"]) / mrc1.xwidth

                ind = mrc2.xdim * mrc2.ydim * z + mrc2.xdim * y + x

                # check density

                if (
                    pos[0] < 0
                    or pos[1] < 0
                    or pos[2] < 0
                    or pos[0] >= mrc1.xdim
                    or pos[1] >= mrc1.ydim
                    or pos[2] >= mrc1.zdim
                ):
                    mrc2.dens[ind] = 0.0
                    mrc2.vec[x][y][z][0] = 0.0
                    mrc2.vec[x][y][z][1] = 0.0
                    mrc2.vec[x][y][z][2] = 0.0
                    continue

                if mrc1.data[int(pos[0])][int(pos[1])][int(pos[2])] == 0:
                    mrc2.dens[ind] = 0.0
                    mrc2.vec[x][y][z][0] = 0.0
                    mrc2.vec[x][y][z][1] = 0.0
                    mrc2.vec[x][y][z][2] = 0.0
                    continue

                ori[0] = pos[0]
                ori[1] = pos[1]
                ori[2] = pos[2]

                # Start Point
                stp[0] = int(pos[0] - fmaxd)
                stp[1] = int(pos[1] - fmaxd)
                stp[2] = int(pos[2] - fmaxd)

                # set start and end point
                if stp[0] < 0:
                    stp[0] = 0
                if stp[1] < 0:
                    stp[1] = 0
                if stp[2] < 0:
                    stp[2] = 0

                endp[0] = int(pos[0] + fmaxd + 1)
                endp[1] = int(pos[1] + fmaxd + 1)
                endp[2] = int(pos[2] + fmaxd + 1)

                if endp[0] >= mrc1.xdim:
                    endp[0] = mrc1.xdim
                if endp[1] >= mrc1.ydim:
                    endp[1] = mrc1.ydim
                if endp[2] >= mrc1.zdim:
                    endp[2] = mrc1.zdim

                # setup for numba acc
                stp_t = List()
                endp_t = List()
                pos_t = List()
                [stp_t.append(x) for x in stp]
                [endp_t.append(x) for x in endp]
                [pos_t.append(x) for x in pos]

                # compute the total density
                dtotal, pos2 = calc(stp_t, endp_t, pos_t, mrc1.data, fsiv)

                mrc2.dens[ind] = dtotal
                mrc2.data[x][y][z] = dtotal

                if dtotal == 0:
                    mrc2.vec[x][y][z][0] = 0.0
                    mrc2.vec[x][y][z][1] = 0.0
                    mrc2.vec[x][y][z][2] = 0.0
                    continue

                rd = 1.0 / dtotal

                pos2[0] *= rd
                pos2[1] *= rd
                pos2[2] *= rd

                tmpcd[0] = pos2[0] - pos[0]
                tmpcd[1] = pos2[1] - pos[1]
                tmpcd[2] = pos2[2] - pos[2]

                dvec = math.sqrt(tmpcd[0] ** 2 + tmpcd[1] ** 2 + tmpcd[2] ** 2)

                if dvec == 0:
                    dvec = 1.0

                rdvec = 1.0 / dvec

                mrc2.vec[x][y][z][0] = tmpcd[0] * rdvec
                mrc2.vec[x][y][z][1] = tmpcd[1] * rdvec
                mrc2.vec[x][y][z][2] = tmpcd[2] * rdvec

                dsum += dtotal
                Nact += 1

    print("#End LDP")
    print(dsum)
    print(Nact)

    mrc2.dsum = dsum
    mrc2.Nact = Nact
    mrc2.ave = dsum / float(Nact)
    mrc2.std = np.linalg.norm(mrc2.dens[mrc2.dens > 0])
    mrc2.std_norm_ave = np.linalg.norm(mrc2.dens[mrc2.dens > 0] - mrc2.ave)

    print("#MAP AVE={ave} STD={std} STD_norm={std_norm}".format(ave=mrc2.ave, std=mrc2.std, std_norm=mrc2.std_norm_ave))
    # return False
    return mrc2

def rot_pos(vec, angle, inv=False):
    r = R.from_euler("zyx", angle, degrees=True)
    if inv:
        r = r.inv()
    rotated_vec = r.apply(vec)
    return rotated_vec

@numba.jit(nopython=True)
def rot_pos_mtx(mtx, vec):
    mtx = mtx.astype(np.float32)
    vec = vec.astype(np.float32)
    return vec @ mtx

def rot_mrc(orig_mrc_data, orig_mrc_vec, angle):
    dim = orig_mrc_vec.shape[0]

    new_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim),)).T.reshape(-1, 3)

    # set the center
    cent = 0.5 * float(dim)

    # get relative positions from center
    new_pos = new_pos - cent
    # print(new_pos)

    # init the rotation by euler angle
    r = R.from_euler("ZYX", angle, degrees=True)
    mtx = r.as_matrix()
    mtx[np.isclose(mtx, 0, atol=1e-15)] = 0

    # print(mtx)

    old_pos = rot_pos_mtx(np.flip(mtx).T, new_pos) + cent
    
    # print(old_pos)

    combined_arr = np.hstack((old_pos, new_pos))

    # filter values outside the boundaries
    in_bound_mask = (
        (old_pos[:, 0] >= 0)
        & (old_pos[:, 1] >= 0)
        & (old_pos[:, 2] >= 0)
        & (old_pos[:, 0] < dim)
        & (old_pos[:, 1] < dim)
        & (old_pos[:, 2] < dim)
    )

    combined_arr = combined_arr[in_bound_mask]

    # print(combined_arr)

    # conversion indices to int
    combined_arr = combined_arr.astype(np.int32)

    # print(combined_arr.shape)

    index_arr = combined_arr[:, 0:3]

    # print(index_arr)
    # print(np.count_nonzero(orig_mrc_data))

    dens_mask = orig_mrc_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0

    # print(dens_mask.shape)
    # print(dens_mask)

    non_zero_rot_list = combined_arr[dens_mask]

    # print(non_zero_rot_list.shape)
    #     with np.printoptions(threshold=np.inf):
    #         print(non_zero_rot_list[:, 0:3])

    non_zero_vec = orig_mrc_vec[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]

    non_zero_dens = orig_mrc_data[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]

    # print(non_zero_dens)

    # non_zero_dens[:, [2, 0]] = non_zero_dens[:, [0, 2]]
    new_vec = rot_pos_mtx(np.flip(mtx), non_zero_vec)

    # print(new_vec)

    # init new vec and dens array
    new_vec_array = np.zeros_like(orig_mrc_vec)
    new_data_array = np.zeros_like(orig_mrc_data)

    # print(new)
    
    new_ind_arr = (non_zero_rot_list[:, 3:6] + cent).astype(int)

    # fill in the new data
#     for vec, ind, dens in zip(new_vec, new_ind_arr, non_zero_dens):
#         new_vec_array[ind[0]][ind[1]][ind[2]] = vec
#         new_data_array[ind[0]][ind[1]][ind[2]] = dens

    new_vec_array[new_ind_arr[:,0], new_ind_arr[:,1], new_ind_arr[:,2]] = new_vec
    new_data_array[new_ind_arr[:,0], new_ind_arr[:,1], new_ind_arr[:,2]] = non_zero_dens

    return new_vec_array, new_data_array

def ang_to_mtx_ZYX(angle):
    r = R.from_euler("ZYX", angle, degrees=True)
    mtx = r.as_matrix()
    mtx[np.isclose(mtx, 0, atol=1e-15)] = 0
    #return np.flip(mtx).T
    mtx = np.flip(mtx).T
    return mtx.astype(np.float32)

def rot_init_cuda(orig_mrc_data, orig_mrc_vec, angle_comb):
    
    import cupy as cp
    
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    rot_mtx_tensor = cp.array([ang_to_mtx_ZYX(ang) for ang in angle_comb])
    
    dim = orig_mrc_data.shape[0]
    cent = 0.5 * float(dim)
    new_pos_arr = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim))).T.reshape(-1, 3)
    new_pos_arr = new_pos_arr - cent
    new_pos_arr = new_pos_arr.astype(np.float32)
    new_pos_arr_gpu = cp.asarray(new_pos_arr)
    
    new_pos_tensor = cp.repeat(
        new_pos_arr_gpu[cp.newaxis, :, :], len(rot_mtx_tensor), axis=0
    )
    
    old_pos_tensor = cp.einsum("ijk,ikl->ijl", new_pos_tensor, rot_mtx_tensor, optimize=True)
    
    old_pos_tensor =old_pos_tensor + cent
    
    bool_masks = (
        (old_pos_tensor[..., 0] >= 0)
        & (old_pos_tensor[..., 1] >= 0)
        & (old_pos_tensor[..., 2] >= 0)
        & (old_pos_tensor[..., 0] < dim)
        & (old_pos_tensor[..., 1] < dim)
        & (old_pos_tensor[..., 2] < dim)
    )
    
    bool_masks_cpu = bool_masks.get()
    
    combined_pos = np.dstack((old_pos_tensor.get(), new_pos_tensor.get())).astype(np.int32)
    
    new_pos_tensor = None
    old_pos_tensor = None
    bool_masks = None
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    
    rot_vec_dict = {}
    rot_data_dict = {}
    
    for angle, mask, pos in zip(angle_comb, bool_masks_cpu, combined_pos):
        pos = pos[mask]
        index_arr = pos[:, 0:3]
        dens_mask = orig_mrc_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
        non_zero_rot_list = pos[dens_mask]
        non_zero_vec = orig_mrc_vec[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
        non_zero_dens = orig_mrc_data[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
        new_vec = non_zero_vec @ np.flip(ang_to_mtx_ZYX(angle)).T
        
        # init new vec and dens array
        new_vec_array = np.zeros_like(orig_mrc_vec)
        new_data_array = np.zeros_like(orig_mrc_data)
        
        new_ind_arr = (non_zero_rot_list[:, 3:6] + cent).astype(int)
        new_vec_array[new_ind_arr[:,0], new_ind_arr[:,1], new_ind_arr[:,2]] = new_vec
        new_data_array[new_ind_arr[:,0], new_ind_arr[:,1], new_ind_arr[:,2]] = non_zero_dens
        
        rot_vec_dict[tuple(angle)] = new_vec_array
        rot_data_dict[tuple(angle)] = new_data_array
    
    return rot_vec_dict, rot_data_dict


def find_best_trans_list(input_list):
    
    sum_arr = np.zeros_like(input_list[0])
    for arr in input_list:
        sum_arr = sum_arr + arr
    best = np.amax(sum_arr)
    trans = np.unravel_index(sum_arr.argmax(), sum_arr.shape)
    
    return best, trans

def find_best_trans_list_prob(input_list):
    
    sum_arr = np.zeros_like(input_list[0])
    #for arr in input_list:
    dot_array=input_list[0]+input_list[1]+input_list[2]
    #avg_dot=np.sum(dot_array)/(dot_array.shape[0])
    ave_dot = np.mean(dot_array)
    std_dot = np.std(dot_array)
    dot_array_z=(dot_array-ave_dot)/std_dot
    
    prob_array=input_list[3]+input_list[4]+input_list[5]+input_list[6]
    #avg_dot=np.sum(dot_array)/(dot_array.shape[0])
    ave_prob = np.mean(prob_array)
    std_prob = np.std(prob_array)
    prob_array_z=(prob_array-ave_prob)/std_prob
    
    
    #sum_arr = sum_arr+input_list[0]+input_list[1]+input_list[2]+(input_list[3] + input_list[4] + input_list[5] + input_list[6])/5000000.00
    sum_arr = sum_arr+dot_array_z+prob_array_z
    prob_arr=input_list[3] + input_list[4] + input_list[5] + input_list[6]
    best = np.amax(sum_arr)
    best_prob=np.amax(prob_array_z)
    trans = np.unravel_index(sum_arr.argmax(), sum_arr.shape)
    
    return best, trans, best_prob


def fft_search_score_trans(target_X, target_Y, target_Z, search_vec, a, b, c, fft_object, ifft_object):
    x2 = copy.deepcopy(search_vec[..., 0])
    y2 = copy.deepcopy(search_vec[..., 1])
    z2 = copy.deepcopy(search_vec[..., 2])

    X2 = np.zeros_like(target_X)
    np.copyto(a, x2)
    np.copyto(X2, fft_object(a))
    dot_X = target_X * X2
    np.copyto(b, dot_X)
    dot_x = np.zeros_like(x2)
    np.copyto(dot_x, ifft_object(b))

    Y2 = np.zeros_like(target_Y)
    np.copyto(a, y2)
    np.copyto(Y2, fft_object(a))
    dot_Y = target_Y * Y2
    np.copyto(b, dot_Y)
    dot_y = np.zeros_like(y2)
    np.copyto(dot_y, ifft_object(b))

    Z2 = np.zeros_like(target_Z)
    np.copyto(a, z2)
    np.copyto(Z2, fft_object(a))
    dot_Z = target_Z * Z2
    np.copyto(b, dot_Z)
    dot_z = np.zeros_like(z2)
    np.copyto(dot_z, ifft_object(b))

    #         if tuple(angle) == (0, 0, 30):
    #             XX = [x12, y12, z12]

    #         X2 = np.fft.rfftn(x2)
    #         X12 = X1 * X2
    #         x12 = np.fft.irfftn(X12, norm="forward")

    # #         if (tuple(angle) == (0,0,30)):
    # #             XX = [X12,x12]

    #         Y2 = np.fft.rfftn(y2)
    #         Y12 = Y1 * Y2
    #         y12 = np.fft.irfftn(Y12, norm="forward")

    #         Z2 = np.fft.rfftn(z2)
    #         Z12 = Z1 * Z2
    #         z12 = np.fft.irfftn(Z12, norm="forward")

    return find_best_trans_list([dot_x, dot_y, dot_z])

def fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object):
    dot_product_list = []
    for target_complex, query_real in zip(target_list, query_list):
        
        query_complex = np.zeros_like(target_complex)
        np.copyto(a, query_real)
        np.copyto(query_complex, fft_object(a))
        dot_complex = target_complex * query_complex
        np.copyto(b, dot_complex)
        dot_real = np.zeros_like(query_real)
        np.copyto(dot_real, ifft_object(b))
        
        dot_product_list.append(dot_real)
        
    return dot_product_list

def fft_search_score_trans_1d(target_X, search_data, a, b, fft_object, ifft_object, mode, ave=None):

    x2 = copy.deepcopy(search_data)

    if mode == "Overlap":
        x2 = np.where(x2 > 0, 1.0, 0.0)
    elif mode == "CC":
        x2 = np.where(x2 > 0, x2, 0.0)
    elif mode == "PCC":
        x2 = np.where(x2 > 0, x2 - ave, 0.0)
    elif mode == "Laplacian":
        weights = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, -6.0, 1.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        x2 = convolve(search_data, weights, mode="constant")
        #x2 = correlate(x2, weights, mode="constant")

    X2 = np.zeros_like(target_X)
    np.copyto(a, x2)
    np.copyto(X2, fft_object(a))
    dot_X = target_X * X2
    np.copyto(b, dot_X)
    dot_x = np.zeros_like(x2)
    np.copyto(dot_x, ifft_object(b))

    return find_best_trans_list([dot_x])

def search_map_fft(mrc_target, mrc_search, TopN=10, ang=30, mode="VecProduct", is_eval_mode=False):

    if is_eval_mode:
        print("#For Evaluation Mode")
        print("#Please use the same coordinate system and map size for map1 and map2.")
        print("#Example:")
        print("#In Chimera command line: open map1 and map2 as #0 and #1, then type")
        print("#> open map1.mrc")
        print("#> open map2.mrc")
        print("#> vop #1 resample onGrid #0")
        print("#> volume #2 save new.mrc")
        print("#Chimera will generate the resampled map2.mrc as new.mrc")
        return

    #     x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    #     y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
    #     z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    # init the target map vectors
    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])

    if mode == "VecProduct":
        y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
        z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    # Postprocessing for other modes
    if mode == "Overlap":
        x1 = np.where(mrc_target.data > 0, 1.0, 0.0)
    elif mode == "CC":
        x1 = np.where(mrc_target.data > 0, mrc_target.data, 0.0)
    elif mode == "PCC":
        x1 = np.where(mrc_target.data > 0, mrc_target.data - mrc_target.ave, 0.0)
    elif mode == "Laplacian":
        weights = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, -6.0, 1.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        x1 = convolve(mrc_target.data, weights, mode="constant")
        #x1 = correlate(mrc_target.data, weights, mode="constant")

    d3 = mrc_target.xdim ** 3

    rd3 = 1.0 / d3

    X1 = np.fft.rfftn(x1)
    X1 = np.conj(X1)

    if mode == "VecProduct":
        Y1 = np.fft.rfftn(y1)
        Y1 = np.conj(Y1)
        Z1 = np.fft.rfftn(z1)
        Z1 = np.conj(Z1)

    x_angle = []
    y_angle = []
    z_angle = []

    i = 0
    while i < 360:
        x_angle.append(i)
        y_angle.append(i)
        i += ang

    i = 0
    while i <= 180:
        z_angle.append(i)
        i += ang

    angle_comb = np.array(np.meshgrid(x_angle, y_angle, z_angle)).T.reshape(-1, 3)
    
#     rot_vec_dict, rot_data_dict = rot_init_cuda(mrc_search.data, mrc_search.vec, angle_comb)

    rot_vec_dict = {}
    rot_data_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() + 4) as executor:
        trans_vec = {executor.submit(rot_mrc, mrc_search.data, mrc_search.vec, angle,): angle for angle in angle_comb}
        for future in concurrent.futures.as_completed(trans_vec):
            angle = trans_vec[future]
            rot_vec_dict[tuple(angle)] = future.result()[0]
            rot_data_dict[tuple(angle)] = future.result()[1]

    # fftw plans
    a = pyfftw.empty_aligned((x1.shape), dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned((x1.shape), dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    angle_score = []

    for angle in tqdm(angle_comb, desc="FFT Process"):
        rot_mrc_vec = rot_vec_dict[tuple(angle)]
        rot_mrc_data = rot_data_dict[tuple(angle)]

        if mode == "VecProduct":
            
            x2 = copy.deepcopy(rot_mrc_vec[..., 0])
            y2 = copy.deepcopy(rot_mrc_vec[..., 1])
            z2 = copy.deepcopy(rot_mrc_vec[..., 2])
                    
            target_list = [X1, Y1, Z1]
            query_list = [x2, y2, z2]
            
            fft_result_list = fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object)
                    
            best, trans = find_best_trans_list(fft_result_list)
            
        else:
            best, trans = fft_search_score_trans_1d(
                X1, rot_mrc_data, a, b, fft_object, ifft_object, mode, mrc_target.ave
            )
            if mode == "CC":
                rstd2 = 1.0 / mrc_target.std ** 2
                best = best * rstd2
            if mode == "PCC":
                rstd3 = 1.0 / mrc_target.std_norm_ave ** 2
                best = best * rstd3

        angle_score.append([tuple(angle), best * rd3, trans])

    # calculate the ave and std
    score_arr = np.array([row[1] for row in angle_score])
    ave = np.mean(score_arr)
    std = np.std(score_arr)
    print("Std= " + str(std) + " Ave= " + str(ave))

    # sort the list and get topN
    sorted_topN = sorted(angle_score, key=lambda x: x[1], reverse=True)[:TopN]

    for x in sorted_topN:
        print(x)

    refined_score = []  
    if ang > 5.0:
        
        # setup all the angles for refinement
        # initialize the refinement list by ±5 degrees
        refine_ang_list = []
        for t_mrc in sorted_topN: 
            curr_ang_arr = np.array(
                np.meshgrid(
                    [t_mrc[0][0] - 5, t_mrc[0][0], t_mrc[0][0] + 5],
                    [t_mrc[0][1] - 5, t_mrc[0][1], t_mrc[0][1] + 5],
                    [t_mrc[0][2] - 5, t_mrc[0][2], t_mrc[0][2] + 5],
                )
            ).T.reshape(-1, 3)
            refine_ang_list.append(curr_ang_arr)
        
        refine_ang_arr = np.concatenate(refine_ang_list, axis=0)
        print(refine_ang_arr.shape)
        
        # rotate the mrc vector and data according to the list (multi-threaded)
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() + 4) as executor:
            trans_vec = {executor.submit(rot_mrc, mrc_search.data, mrc_search.vec, angle,): angle for angle in refine_ang_arr}
            for future in concurrent.futures.as_completed(trans_vec):
                angle = trans_vec[future]
                rot_vec_dict[tuple(angle)] = future.result()[0]
                rot_data_dict[tuple(angle)] = future.result()[1]
                
        for angle in tqdm(refine_ang_arr, desc="Refine FFT Process"):
            
            rot_mrc_vec = rot_vec_dict[tuple(angle)]
            rot_mrc_data = rot_data_dict[tuple(angle)]
            
            if mode == "VecProduct":
                x2 = copy.deepcopy(rot_mrc_vec[..., 0])
                y2 = copy.deepcopy(rot_mrc_vec[..., 1])
                z2 = copy.deepcopy(rot_mrc_vec[..., 2])
                    
                target_list = [X1, Y1, Z1]
                query_list = [x2, y2, z2]
            
                fft_result_list = fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object)
                best, trans = find_best_trans_list(fft_result_list)
            
            else:
                best, trans = fft_search_score_trans_1d(
                    X1, rot_mrc_data, a, b, fft_object, ifft_object, mode, mrc_target.ave
                )
                if mode == "CC":
                    rstd2 = 1.0 / mrc_target.std ** 2
                    best = best * rstd2
                if mode == "PCC":
                    rstd3 = 1.0 / mrc_target.std_norm_ave ** 2
                    best = best * rstd3
        
            refined_score.append([tuple(angle), best * rd3, trans, rot_mrc_vec, rot_mrc_data])
            
        refined_list = sorted(refined_score, key=lambda x: x[1], reverse=True)[:TopN]
    
    else:
        refined_list = sorted_topN
    
    # Save the results to file
    for i, t_mrc in enumerate(refined_list):
        
        # calculate the scores
        print("R=" + str(t_mrc[0]) + " T=" + str(t_mrc[2]))
        sco = get_dot_score(
            mrc_target.data,
            t_mrc[4],
            mrc_target.vec,
            t_mrc[3],
            t_mrc[2],
            mrc_target.ave,
            mrc_search.ave,
            mrc_target.std,
            mrc_search.std,
            mrc_target.std_norm_ave,
            mrc_search.std_norm_ave,
        )
        
        # Write result to PDB files
        show_vec(mrc_target.orig, t_mrc[3], t_mrc[4], sco, mrc_search.xwidth, t_mrc[2], "model_" + str(i + 1) + ".pdb")

    return refined_list

def search_map_fft_prob(mrc_P1, mrc_P2, mrc_P3, mrc_P4, mrc_target, mrc_search,mrc_search_p1,mrc_search_p2,mrc_search_p3,mrc_search_p4, TopN=10, ang=10, mode="VecProduct", is_eval_mode=False):

    if is_eval_mode:
        print("#For Evaluation Mode")
        print("#Please use the same coordinate system and map size for map1 and map2.")
        print("#Example:")
        print("#In Chimera command line: open map1 and map2 as #0 and #1, then type")
        print("#> open map1.mrc")
        print("#> open map2.mrc")
        print("#> vop #1 resample onGrid #0")
        print("#> volume #2 save new.mrc")
        print("#Chimera will generate the resampled map2.mrc as new.mrc")
        return

    #     x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    #     y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
    #     z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    # init the target map vectors
    #does this part need to be changed?
    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    p1 = copy.deepcopy(mrc_P1.data)
    p2 = copy.deepcopy(mrc_P2.data)
    p3 = copy.deepcopy(mrc_P3.data)
    p4 = copy.deepcopy(mrc_P4.data)


    if mode == "VecProduct":
        y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
        z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    # Postprocessing for other modes
    if mode == "Overlap":
        x1 = np.where(x1 > 0, 1.0, 0.0)
    elif mode == "CC":
        x1 = np.where(x1 > 0, x1, 0.0)
    elif mode == "PCC":
        x1 = np.where(x1 > 0, x1 - mrc_target.ave, 0.0)
    elif mode == "Laplacian":
        weights = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, -6.0, 1.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        )
        x1 = convolve(x1, weights, mode="constant")

    d3 = mrc_target.xdim ** 3

    rd3 = 1.0 / d3
    
    
    X1 = np.fft.rfftn(x1)
    X1 = np.conj(X1)
    P1 = np.fft.rfftn(p1)
    P1 = np.conj(P1)
    P2 = np.fft.rfftn(p2)
    P2 = np.conj(P2)
    P3 = np.fft.rfftn(p3)
    P3 = np.conj(P3)
    P4 = np.fft.rfftn(p4)
    P4 = np.conj(P4)
    
    #P4 = 1 - P2 - P3 - P1

    if mode == "VecProduct":
        Y1 = np.fft.rfftn(y1)
        Y1 = np.conj(Y1)
        Z1 = np.fft.rfftn(z1)
        Z1 = np.conj(Z1)

    x_angle = []
    y_angle = []
    z_angle = []

    i = 0
    while i < 360:
        x_angle.append(i)
        y_angle.append(i)
        i += ang

    i = 0
    while i <= 180:
        z_angle.append(i)
        i += ang

    angle_comb = np.array(np.meshgrid(x_angle, y_angle, z_angle)).T.reshape(-1, 3)

    rot_vec_dict = {}
    rot_data_dict = {}
    #rot_vec_dict_p1 = {}
    rot_data_dict_p1 = {}
    #rot_vec_dict_p2 = {}
    rot_data_dict_p2 = {}
    #rot_vec_dict_p3 = {}
    rot_data_dict_p3 = {}
    #rot_vec_dict_p4 = {}
    rot_data_dict_p4 = {}
    

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() + 4) as executor:
        trans_vec = {executor.submit(rot_mrc_prob, mrc_search.data, mrc_search.vec, mrc_search_p1.data, mrc_search_p2.data, mrc_search_p3.data, mrc_search_p4.data, angle,): angle for angle in angle_comb}
        #trans_vec_p1 = {executor.submit(rot_mrc, mrc_search_p1.data, mrc_search_p1.vec, angle,): angle for angle in angle_comb}
        #trans_vec_p2 = {executor.submit(rot_mrc, mrc_search_p2.data, mrc_search_p2.vec, angle,): angle for angle in angle_comb}
        #trans_vec_p3 = {executor.submit(rot_mrc, mrc_search_p3.data, mrc_search_p3.vec, angle,): angle for angle in angle_comb}
        #trans_vec_p4 = {executor.submit(rot_mrc, mrc_search_p4.data, mrc_search_p4.vec, angle,): angle for angle in angle_comb}
        for future in concurrent.futures.as_completed(trans_vec):
            angle = trans_vec[future]
            rot_vec_dict[tuple(angle)] = future.result()[0]
            rot_data_dict[tuple(angle)] = future.result()[1]
            rot_data_dict_p1[tuple(angle)] = future.result()[2]
            rot_data_dict_p2[tuple(angle)] = future.result()[3]
            rot_data_dict_p3[tuple(angle)] = future.result()[4]
            rot_data_dict_p4[tuple(angle)] = future.result()[5]

    #     for angle in tqdm(angle_comb, desc="Rotation"):
    #         rot_result = rot_mrc(
    #             mrc_search.data,
    #             mrc_search.vec,
    #             angle,
    #         )
    #         mrc_angle_dict[tuple(angle)] = rot_result

    # fftw plans
    a = pyfftw.empty_aligned((x1.shape), dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned((x1.shape), dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    angle_score = []

    for angle in tqdm(angle_comb, desc="FFT"):
        rot_mrc_vec = rot_vec_dict[tuple(angle)]
        rot_mrc_data = rot_data_dict[tuple(angle)]
        #rot_mrc_vec_p1 = rot_vec_dict_p1[tuple(angle)]
        rot_mrc_data_p1 = rot_data_dict_p1[tuple(angle)]
        #rot_mrc_vec_p2 = rot_vec_dict_p2[tuple(angle)]
        rot_mrc_data_p2 = rot_data_dict_p2[tuple(angle)]
        #rot_mrc_vec_p3 = rot_vec_dict_p3[tuple(angle)]
        rot_mrc_data_p3 = rot_data_dict_p3[tuple(angle)]
        #rot_mrc_vec_p4 = rot_vec_dict_p4[tuple(angle)]
        rot_mrc_data_p4 = rot_data_dict_p4[tuple(angle)]
        
      
        if mode == "VecProduct":
            
            x2 = copy.deepcopy(rot_mrc_vec[..., 0])
            y2 = copy.deepcopy(rot_mrc_vec[..., 1])
            z2 = copy.deepcopy(rot_mrc_vec[..., 2])
            p21 = copy.deepcopy(rot_mrc_data_p1)
            p22 = copy.deepcopy(rot_mrc_data_p2)
            p23 = copy.deepcopy(rot_mrc_data_p3)
            p24 = copy.deepcopy(rot_mrc_data_p4)
            
            #p24 = 1 - (p21 + p22 + p23)
                    
            target_list = [X1, Y1, Z1, P1, P2, P3, P4]
            query_list = [x2, y2, z2, p21, p22, p23, p24]
            
            fft_result_list = fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object)
                    
            best, trans, best_prob = find_best_trans_list_prob(fft_result_list)
            
        else:
            best, trans = fft_search_score_trans_1d(
                X1, rot_mrc_data, a, b, fft_object, ifft_object, mode, mrc_target.ave
            )
            if mode == "CC":
                rstd2 = 1.0 / mrc_target.std ** 2
                best = best * rstd2
            if mode == "PCC":
                rstd3 = 1.0 / mrc_target.std_norm_ave ** 2
                best = best * rstd3

        angle_score.append([tuple(angle), best * rd3, trans])

    # calculate the ave and std
    score_arr = np.array([row[1] for row in angle_score])
    ave = np.mean(score_arr)
    std = np.std(score_arr)
    print("Std= " + str(std) + " Ave= " + str(ave))

    # sort the list and get topN
    sorted_topN = sorted(angle_score, key=lambda x: x[1], reverse=True)[:TopN]

    for x in sorted_topN:
        print(x)

    refined_score = []
    if ang > 5.0:
        for t_mrc in sorted_topN:
            ang_list = np.array(
                np.meshgrid(
                    [t_mrc[0][0] - 5, t_mrc[0][0], t_mrc[0][0] + 5],
                    [t_mrc[0][1] - 5, t_mrc[0][1], t_mrc[0][1] + 5],
                    [t_mrc[0][2] - 5, t_mrc[0][2], t_mrc[0][2] + 5],
                )
            ).T.reshape(-1, 3)
            # print(ang_list)
            for ang in ang_list:
                rotated = rot_mrc_prob(mrc_search.data, mrc_search.vec, mrc_search_p1.data, mrc_search_p2.data, mrc_search_p3.data, mrc_search_p4.data, ang)
                rotated_vec=rotated[0]
                rotated_data=rotated[1]
                rotated_data_p1=rotated[2]
                rotated_data_p2=rotated[3]
                rotated_data_p3=rotated[4]
                rotated_data_p4=rotated[5]
                if mode == "VecProduct":
                    
                    x2 = copy.deepcopy(rotated_vec[..., 0])
                    y2 = copy.deepcopy(rotated_vec[..., 1])
                    z2 = copy.deepcopy(rotated_vec[..., 2])
                    p21 = copy.deepcopy(rotated[2])
                    p22 = copy.deepcopy(rotated[3])
                    p23 = copy.deepcopy(rotated[4])
                    p24 = copy.deepcopy(rotated[5])
                    
                    target_list = [X1, Y1, Z1, P1, P2, P3, P4]
                    query_list = [x2, y2, z2, p21, p22, p23, p24]
                    
                    fft_result_list = fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object)
                    
                    best, trans, best_prob = find_best_trans_list_prob(fft_result_list)
                    
                    # best, trans = fft_search_score_trans(X1, Y1, Z1, rotated_vec, a, b, c, fft_object, ifft_object)
                else:
                    best, trans = fft_search_score_trans_1d(
                        X1, rotated_data, a, b, fft_object, ifft_object, mode, ave=mrc_target.ave
                    )
                    if mode == "CC":
                        rstd2 = 1.0 / mrc_target.std ** 2
                        best = best * rstd2
                    if mode == "PCC":
                        rstd3 = 1.0 / mrc_target.std_norm_ave ** 2
                        best = best * rstd3

                refined_score.append([tuple(ang), best * rd3, trans, rotated_vec, rotated_data, best_prob * rd3])

        refined_list = sorted(refined_score, key=lambda x: x[1], reverse=True)[:TopN]
    else:
        refined_list = sorted_topN

    for i, t_mrc in enumerate(refined_list):
        print("R=" + str(t_mrc[0]) + " T=" + str(t_mrc[2]))
        sco = get_score(
            mrc_target.data,
            t_mrc[4],
            mrc_target.vec,
            t_mrc[3],
            t_mrc[2],
            mrc_target.ave,
            mrc_search.ave,
            mrc_target.std,
            mrc_search.std,
            mrc_target.std_norm_ave,
            mrc_search.std_norm_ave,
            sco_prob_added=t_mrc[1],
            sco_prob_only=t_mrc[5],
        )
        
        # Write result to PDB files
        show_vec(mrc_target.orig, t_mrc[3], t_mrc[4], sco, mrc_search.xwidth, t_mrc[2], "model_" + str(i + 1) + ".pdb")

    #     num_jobs = math.ceil(360 / ang) * math.ceil(360 / ang) * (180 // ang + 1)

    return refined_list

def show_vec(origin, sampled_mrc_vec, sampled_mrc_data, sampled_mrc_score, sample_width, trans, name):

    dim = sampled_mrc_data.shape[0]

    origin = np.array([origin["x"], origin["y"], origin["z"]])
    trans = np.array(trans)

    if trans[0] > 0.5 * dim:
        trans[0] -= dim
    if trans[1] > 0.5 * dim:
        trans[1] -= dim
    if trans[2] > 0.5 * dim:
        trans[2] -= dim

    add = origin - trans * sample_width

    natm = 1
    nres = 1

    pdb_file = open(name, "w")
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):

                if sampled_mrc_data[x][y][z] != 0.0:
                    tmp = np.array([x, y, z])
                    tmp = tmp * sample_width + add
                    atom_header = "ATOM{:>7d}  CA  ALA{:>6d}    ".format(natm, nres)
                    atom_content = "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format(
                        tmp[0], tmp[1], tmp[2], 1.0, sampled_mrc_score[x][y][z]
                    )
                    pdb_file.write(atom_header + atom_content + "\n")
                    natm += 1

                    tmp = np.array([x, y, z])
                    tmp = (tmp + sampled_mrc_vec[x][y][z]) * sample_width + add
                    atom_header = "ATOM{:>7d}  CB  ALA{:>6d}    ".format(natm, nres)
                    atom_content = "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format(
                        tmp[0], tmp[1], tmp[2], 1.0, sampled_mrc_score[x][y][z]
                    )
                    pdb_file.write(atom_header + atom_content + "\n")
                    natm += 1

                    nres += 1

@numba.jit(nopython=False, forceobj=True)
def fastVEC_prob(mrc1, mrc2, dreso=16.0):

    xydim = mrc1.xdim * mrc1.ydim
    Ndata = mrc2.xdim * mrc2.ydim * mrc2.zdim

    print(len(mrc2.dens))

    print("#Start VEC")
    gstep = mrc1.xwidth
    fs = (dreso / gstep) * 0.5
    fs = fs ** 2
    fsiv = 1.0 / fs
    fmaxd = (dreso / gstep) * 2.0
    print("#maxd= {fmaxd}".format(fmaxd=fmaxd))
    print("#fsiv= " + str(fsiv))

    dsum = 0.0
    Nact = 0

    list_d = []

    for x in tqdm(range(mrc2.xdim)):
        for y in range(mrc2.ydim):
            for z in range(mrc2.zdim):
                stp = [0] * 3
                endp = [0] * 3
                ind2 = 0
                ind = 0

                pos = [0.0] * 3
                pos2 = [0.0] * 3
                ori = [0.0] * 3

                tmpcd = [0.0] * 3

                v, dtotal, rd = 0.0, 0.0, 0.0

                pos[0] = (x * mrc2.xwidth + mrc2.orig["x"] - mrc1.orig["x"]) / mrc1.xwidth
                pos[1] = (y * mrc2.xwidth + mrc2.orig["y"] - mrc1.orig["y"]) / mrc1.xwidth
                pos[2] = (z * mrc2.xwidth + mrc2.orig["z"] - mrc1.orig["z"]) / mrc1.xwidth

                ind = mrc2.xdim * mrc2.ydim * z + mrc2.xdim * y + x

                # check density

                if (
                    pos[0] < 0
                    or pos[1] < 0
                    or pos[2] < 0
                    or pos[0] >= mrc1.xdim
                    or pos[1] >= mrc1.ydim
                    or pos[2] >= mrc1.zdim
                ):
                    mrc2.dens[ind] = 0.0
                    #mrc2.vec[x][y][z][0] = 0.0
                    #mrc2.vec[x][y][z][1] = 0.0
                    #mrc2.vec[x][y][z][2] = 0.0
                    continue

                if mrc1.data[int(pos[0])][int(pos[1])][int(pos[2])] == 0:
                    mrc2.dens[ind] = 0.0
                    #mrc2.vec[x][y][z][0] = 0.0
                    #mrc2.vec[x][y][z][1] = 0.0
                    #mrc2.vec[x][y][z][2] = 0.0
                    continue

                ori[0] = pos[0]
                ori[1] = pos[1]
                ori[2] = pos[2]

                # Start Point
                stp[0] = int(pos[0] - fmaxd)
                stp[1] = int(pos[1] - fmaxd)
                stp[2] = int(pos[2] - fmaxd)

                # set start and end point
                if stp[0] < 0:
                    stp[0] = 0
                if stp[1] < 0:
                    stp[1] = 0
                if stp[2] < 0:
                    stp[2] = 0

                endp[0] = int(pos[0] + fmaxd + 1)
                endp[1] = int(pos[1] + fmaxd + 1)
                endp[2] = int(pos[2] + fmaxd + 1)

                if endp[0] >= mrc1.xdim:
                    endp[0] = mrc1.xdim
                if endp[1] >= mrc1.ydim:
                    endp[1] = mrc1.ydim
                if endp[2] >= mrc1.zdim:
                    endp[2] = mrc1.zdim

                # setup for numba acc
                stp_t = List()
                endp_t = List()
                pos_t = List()
                [stp_t.append(x) for x in stp]
                [endp_t.append(x) for x in endp]
                [pos_t.append(x) for x in pos]

                # compute the total density
                dtotal, pos2 = calc(stp_t, endp_t, pos_t, mrc1.data, fsiv)

                mrc2.dens[ind] = dtotal
                mrc2.data[x][y][z] = dtotal

                ''''if dtotal == 0:
                    mrc2.vec[x][y][z][0] = 0.0
                    mrc2.vec[x][y][z][1] = 0.0
                    mrc2.vec[x][y][z][2] = 0.0
                    continue'''

                ''''rd = 1.0 / dtotal

                pos2[0] *= rd
                pos2[1] *= rd
                pos2[2] *= rd

                tmpcd[0] = pos2[0] - pos[0]
                tmpcd[1] = pos2[1] - pos[1]
                tmpcd[2] = pos2[2] - pos[2]

                dvec = math.sqrt(tmpcd[0] ** 2 + tmpcd[1] ** 2 + tmpcd[2] ** 2)

                if dvec == 0:
                    dvec = 1.0

                rdvec = 1.0 / dvec'''

                #mrc2.vec[x][y][z][0] = tmpcd[0] * rdvec
                #mrc2.vec[x][y][z][1] = tmpcd[1] * rdvec
                #mrc2.vec[x][y][z][2] = tmpcd[2] * rdvec

                dsum += dtotal
                Nact += 1

    print("#End LDP")
    print(dsum)
    print(Nact)

    mrc2.dsum = dsum
    mrc2.Nact = Nact
    mrc2.ave = dsum / float(Nact)
    mrc2.std = np.linalg.norm(mrc2.dens[mrc2.dens > 0])
    mrc2.std_norm_ave = np.linalg.norm(mrc2.dens[mrc2.dens > 0] - mrc2.ave)

    print("#MAP AVE={ave} STD={std} STD_norm={std_norm}".format(ave=mrc2.ave, std=mrc2.std, std_norm=mrc2.std_norm_ave))
    # return False
    return mrc2

def rot_mrc_prob(orig_mrc_data, orig_mrc_vec,mrc_search_p1_data, mrc_search_p2_data, mrc_search_p3_data, mrc_search_p4_data,  angle):
    dim = orig_mrc_vec.shape[0]

    new_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim),)).T.reshape(-1, 3)

    cent = 0.5 * float(dim)
    new_pos = new_pos - cent

    r = R.from_euler("ZYX", angle, degrees=True)
    mtx = r.as_matrix()
    mtx[np.isclose(mtx, 0, atol=1e-15)] = 0

    old_pos = rot_pos_mtx(np.flip(mtx).T, new_pos) + cent

    combined_arr = np.hstack((old_pos, new_pos))
    
    in_bound_mask = (
        (old_pos[:, 0] >= 0)
        & (old_pos[:, 1] >= 0)
        & (old_pos[:, 2] >= 0)
        & (old_pos[:, 0] < dim)
        & (old_pos[:, 1] < dim)
        & (old_pos[:, 2] < dim)
    )

    combined_arr = combined_arr[in_bound_mask]

    combined_arr = combined_arr.astype(np.int32)

    index_arr = combined_arr[:, 0:3]

  
    dens_mask = orig_mrc_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    dens_mask_p1 = mrc_search_p1_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    dens_mask_p2 = mrc_search_p2_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    dens_mask_p3 = mrc_search_p3_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    dens_mask_p4 = mrc_search_p4_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0

    non_zero_rot_list = combined_arr[dens_mask]
    non_zero_rot_list_p1 = combined_arr[dens_mask_p1]
    non_zero_rot_list_p2 = combined_arr[dens_mask_p2]
    non_zero_rot_list_p3 = combined_arr[dens_mask_p3]
    non_zero_rot_list_p4 = combined_arr[dens_mask_p4]



    non_zero_vec = orig_mrc_vec[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]

    non_zero_dens = orig_mrc_data[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    
    non_zero_dens_p1 = mrc_search_p1_data[non_zero_rot_list_p1[:, 0], non_zero_rot_list_p1[:, 1], non_zero_rot_list_p1[:, 2]]
    
    non_zero_dens_p2 = mrc_search_p2_data[non_zero_rot_list_p2[:, 0], non_zero_rot_list_p2[:, 1], non_zero_rot_list_p2[:, 2]]
    
    non_zero_dens_p3 = mrc_search_p3_data[non_zero_rot_list_p3[:, 0], non_zero_rot_list_p3[:, 1], non_zero_rot_list_p3[:, 2]]
    
    non_zero_dens_p4 = mrc_search_p4_data[non_zero_rot_list_p4[:, 0], non_zero_rot_list_p4[:, 1], non_zero_rot_list_p4[:, 2]]

    new_vec = rot_pos_mtx(np.flip(mtx), non_zero_vec)


    new_vec_array = np.zeros_like(orig_mrc_vec)
    new_data_array = np.zeros_like(orig_mrc_data)
    new_data_array_p1 = np.zeros_like(mrc_search_p1_data)
    new_data_array_p2 = np.zeros_like(mrc_search_p2_data)
    new_data_array_p3 = np.zeros_like(mrc_search_p3_data)
    new_data_array_p4 = np.zeros_like(mrc_search_p4_data)

    for vec, ind, dens in zip(new_vec, (non_zero_rot_list[:, 3:6] + cent).astype(int), non_zero_dens):
        new_vec_array[ind[0]][ind[1]][ind[2]][0] = vec[0]
        new_vec_array[ind[0]][ind[1]][ind[2]][1] = vec[1]
        new_vec_array[ind[0]][ind[1]][ind[2]][2] = vec[2]
        new_data_array[ind[0]][ind[1]][ind[2]] = dens
        
    for ind, dens in zip((non_zero_rot_list_p1[:, 3:6] + cent).astype(int), non_zero_dens_p1):

        new_data_array_p1[ind[0]][ind[1]][ind[2]] = dens
        
    for ind, dens in zip((non_zero_rot_list_p2[:, 3:6] + cent).astype(int), non_zero_dens_p2):
        new_data_array_p2[ind[0]][ind[1]][ind[2]] = dens
        
    for ind, dens in zip((non_zero_rot_list_p3[:, 3:6] + cent).astype(int), non_zero_dens_p3):
        new_data_array_p3[ind[0]][ind[1]][ind[2]] = dens
        
    for ind, dens in zip((non_zero_rot_list_p4[:, 3:6] + cent).astype(int), non_zero_dens_p4):
        new_data_array_p4[ind[0]][ind[1]][ind[2]] = dens

    return new_vec_array, new_data_array, new_data_array_p1, new_data_array_p2, new_data_array_p3, new_data_array_p4

def get_dot_score(
    target_map_data, search_map_data, target_map_vec, search_map_vec, trans, ave1, ave2, std1, std2, pstd1, pstd2
):

    px, py, pz = 0, 0, 0
    dim = target_map_data.shape[0]
    total = 0

    t = np.array(trans)
    if trans[0] > 0.5 * dim:
        t[0] -= dim
    if trans[1] > 0.5 * dim:
        t[1] -= dim
    if trans[2] > 0.5 * dim:
        t[2] -= dim

    target_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim),)).T.reshape(-1, 3)

    search_pos = target_pos + t

    total += np.count_nonzero(target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]])

    combined_arr = np.hstack((target_pos, search_pos))

    combined_arr = combined_arr[
        (combined_arr[:, 3] >= 0)
        & (combined_arr[:, 4] >= 0)
        & (combined_arr[:, 5] >= 0)
        & (combined_arr[:, 3] < dim)
        & (combined_arr[:, 4] < dim)
        & (combined_arr[:, 5] < dim)
    ]

    target_pos = combined_arr[:, 0:3]
    search_pos = combined_arr[:, 3:6]

    d1 = target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]]
    d2 = search_map_data[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]]

    d1 = np.where(d1 <= 0, 0.0, d1)
    d2 = np.where(d2 <= 0, 0.0, d1)

    print(np.sum(d1))
    print(np.sum(d2))

    pd1 = np.where(d1 <= 0, 0.0, d1 - ave1)
    pd2 = np.where(d2 <= 0, 0.0, d2 - ave2)

    cc = np.sum(np.multiply(d1, d2))
    pcc = np.sum(np.multiply(pd1, pd2))

    target_zero_mask = target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]] == 0
    target_non_zero_mask = target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]] > 0
    search_non_zero_mask = search_map_data[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]] > 0
    search_non_zero_count = np.count_nonzero(np.multiply(target_zero_mask, search_non_zero_mask))

    trimmed_target_vec = target_map_vec[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]]
    trimmed_search_vec = search_map_vec[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]]

    total += search_non_zero_count

    sco_arr = np.zeros_like(search_map_data)
    sco = np.einsum("ij,ij->i", trimmed_target_vec, trimmed_search_vec)
    sco_arr[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]] = sco
    sco_sum = np.sum(sco_arr)
    Nm = np.count_nonzero(np.multiply(target_non_zero_mask, search_non_zero_mask))

    print(
        "Overlap= "
        + str(float(Nm) / float(total))
        + " "
        + str(Nm)
        + "/"
        + str(total)
        + " CC= "
        + str(cc / (std1 * std2))
        + " PCC= "
        + str(pcc / (pstd1 * pstd2))
    )
    print("Score=", sco_sum)
    return sco_arr


def get_score(
    target_map_data, search_map_data, target_map_vec, search_map_vec, trans, ave1, ave2, std1, std2, pstd1, pstd2, sco_prob_added, sco_prob_only
):
    px, py, pz = 0, 0, 0
    dim = target_map_data.shape[0]
    total = 0

    t = np.array(trans)
    if trans[0] > 0.5 * dim:
        t[0] -= dim
    if trans[1] > 0.5 * dim:
        t[1] -= dim
    if trans[2] > 0.5 * dim:
        t[2] -= dim
        
    #duplicate this part for all probability maps

    target_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim),)).T.reshape(-1, 3)

    search_pos = target_pos + t

    total += np.count_nonzero(target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]])

    combined_arr = np.hstack((target_pos, search_pos))

    combined_arr = combined_arr[
        (combined_arr[:, 3] >= 0)
        & (combined_arr[:, 4] >= 0)
        & (combined_arr[:, 5] >= 0)
        & (combined_arr[:, 3] < dim)
        & (combined_arr[:, 4] < dim)
        & (combined_arr[:, 5] < dim)
    ]

    target_pos = combined_arr[:, 0:3]
    search_pos = combined_arr[:, 3:6]

    d1 = target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]]
    d2 = search_map_data[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]]
    
    d1 = np.where(d1 <= 0, 0.0, d1)
    d2 = np.where(d2 <= 0, 0.0, d1)

    print(np.sum(d1))
    print(np.sum(d2))

    pd1 = np.where(d1 <= 0, 0.0, d1 - ave1)
    pd2 = np.where(d2 <= 0, 0.0, d2 - ave2)

    cc = np.sum(np.multiply(d1, d2))
    pcc = np.sum(np.multiply(pd1, pd2))

    target_zero_mask = target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]] == 0
    target_non_zero_mask = target_map_data[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]] > 0
    search_non_zero_mask = search_map_data[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]] > 0
    search_non_zero_count = np.count_nonzero(np.multiply(target_zero_mask, search_non_zero_mask))

    trimmed_target_vec = target_map_vec[target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]]
    trimmed_search_vec = search_map_vec[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]]

    total += search_non_zero_count

    sco_arr = np.zeros_like(search_map_data)
    sco = np.einsum("ij,ij->i", trimmed_target_vec, trimmed_search_vec)
    sco_arr[search_pos[:, 0], search_pos[:, 1], search_pos[:, 2]] = sco
    sco_sum = np.sum(sco_arr)
    Nm = np.count_nonzero(np.multiply(target_non_zero_mask, search_non_zero_mask))

    print(
        "Overlap= "
        + str(float(Nm) / float(total))
        + " "
        + str(Nm)
        + "/"
        + str(total)
        + " CC= "
        + str(cc / (std1 * std2))
        + " PCC= "
        + str(pcc / (pstd1 * pstd2))
        + " Scoreplusprob= "
        + str(sco_prob_added)
        + " Scoreprobonly= "
        + str(sco_prob_only)
    )
    print("Score= ", sco_sum)
    return sco_arr