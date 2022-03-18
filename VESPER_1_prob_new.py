# coding: utf-8
import concurrent.futures
import copy
import multiprocessing
import os
import time

import mrcfile
import numba
import numpy as np
import pyfftw
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()


class mrc_obj:
    """A mrc object that represents the density data and statistics of a given mrc file"""

    def __init__(self, path):
        # open the specified mrcfile and read the header information
        mrc = mrcfile.open(path)
        data = mrc.data
        header = mrc.header

        # read and store the voxel widths and dimensions from the header
        self.xdim = int(header.nx)
        self.ydim = int(header.ny)
        self.zdim = int(header.nz)
        self.xwidth = mrc.voxel_size.x
        self.ywidth = mrc.voxel_size.y
        self.zwidth = mrc.voxel_size.z

        # set the center to be the half the dimensions
        self.cent = np.array([self.xdim * 0.5, self.ydim * 0.5, self.zdim * 0.5,
                              ])

        # read and store the origin coordinate from the header
        self.orig = np.array([header.origin.x, header.origin.y, header.origin.z])

        # swap the xz axes of density data array and store in self.data
        self.data = np.swapaxes(copy.deepcopy(data), 0, 2)

        # create 1d representation of the density value by flattening the data array
        self.dens = data.flatten()

        # initialize the vector array to be same shape as data but will all zeros
        self.vec = np.zeros((self.xdim, self.ydim, self.zdim, 3), dtype="float32")

        # initialize all the statistics values
        self.dsum = None  # total density value
        self.Nact = None  # non-zero density voxel count
        self.ave = None  # average density value
        self.std_norm_ave = None  # L2 norm nomalized with average density value
        self.std = None  # unnormalize L2 norm


def mrc_set_vox_size(mrc, th=0.00, voxel_size=7.0):
    """Set the voxel size for the specified mrc_obj

    Args:
        mrc (mrc_obj): [the target mrc_obj to set the voxel size for]
        th (float, optional): preset threshold for density cutoff. Defaults to 0.01.
        voxel_size (float, optional): the granularity of the voxel in terms of angstroms. Defaults to 7.0.

    Returns:
        mrc (mrc_obj): the original mrc_obj
        mrc_new (mrc_obj): a processed mrc_obj
    """

    # if th < 0 add th to all value
    if th < 0:
        mrc.dens = mrc.dens - th
        th = 0.0

    # zero all the values less than threshold
    mrc.dens[mrc.dens < th] = 0.0
    mrc.data[mrc.data < th] = 0.0

    # calculate maximum distance for non-zero entries
    non_zero_index_list = np.array(np.nonzero(mrc.data)).T
    cent_arr = np.array(mrc.cent)
    d2_list = np.linalg.norm(non_zero_index_list - cent_arr, axis=1)
    dmax = max(d2_list)

    print("#dmax=" + str(dmax / mrc.xwidth))
    dmax = dmax * mrc.xwidth

    # set new center
    new_cent = mrc.cent * mrc.xwidth + mrc.orig

    tmp_size = 2 * dmax / voxel_size

    # get the best size suitable for fft operation
    new_xdim = pyfftw.next_fast_len(int(tmp_size))

    # set new origins
    new_orig = new_cent - 0.5 * new_xdim * voxel_size

    # create new mrc object
    mrc_new = copy.deepcopy(mrc)
    mrc_new.orig = new_orig
    mrc_new.xdim = new_xdim
    mrc_new.ydim = new_xdim
    mrc_new.zdim = new_xdim
    mrc_new.cent = new_cent
    mrc_new.xwidth = mrc_new.ywidth = mrc_new.zwidth = voxel_size

    print("Nvox= " + str(mrc_new.xdim) + ", " + str(mrc_new.ydim) + ", " +
          str(mrc_new.zdim))
    print("cent= " + str(new_cent[0]) + ", " + str(new_cent[1]) + ", " +
          str(new_cent[2]))
    print("ori= " + str(new_orig[0]) + ", " + str(new_orig[1]) + ", " +
          str(new_orig[2]))

    return mrc, mrc_new


@numba.jit(nopython=True)
def apply_kern(arr, kernel):
    """Jit compiled function to apply a kernel to a given array (element-wise product).

    Args:
        arr (numpy.array): the array to apply the kernel on
        kernel (numpy.array): the kernel to be applied, should be the same shape as the input arr

    Returns:
        dtotal (float): the total density
        filtered (numpy.array): the filtered array
    """
    filtered = np.multiply(arr, kernel)  # apply guassian kernel
    dtotal = np.sum(filtered)
    return dtotal, filtered


def gkern3d(l=5, sig=1.):
    """
    creates a 3D gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))  # calculate gaussian kernel along 1d axis
    kernel = (gauss[..., None] * gauss)[..., None] * gauss  # out product to produce form 3d kernel
    kernel = kernel / np.sum(kernel)  # Normalization
    return kernel


def fastVEC(mrc_source, mrc_dest, dreso=16.0, calc_vec=True):
    """A function that resample the mrc object to preset voxel size and calculate the vector and other statistics

    Args:
        mrc_source (mrc_obj): The source mrc_obj
        mrc_dest (mrc_obj): The destination mrc_obj
        dreso (float, optional): Gaussian kernel window size. Defaults to 16.0.
        calc_vec (bool, optional): Choose to calculate the vector or not, not required for simulated probability maps. Defaults to True.

    Returns:
        mrc_dest (mrc_obj): converted mrc_obj
    """

    print("#Start VEC")
    gstep = mrc_source.xwidth
    sigma = (dreso / gstep) * 0.3  # calculate the sigma value, 0.3 is an abitrary coefficient
    fmaxd = (dreso / gstep) * 2.0  # calculate filter maximum radius
    print("#maxd= {fmaxd}".format(fmaxd=fmaxd))

    dsum = 0.0  # sum of all calculated density values
    Nact = 0  # non-zero density count

    kernel_length = int(2 * np.ceil(fmaxd)) + 1  # calculate the kernel length
    g_kern = gkern3d(kernel_length, sigma)  # generate kernel

    # iterate over the all the positions in the new grid
    for x in tqdm(range(mrc_dest.xdim)):
        for y in range(mrc_dest.ydim):
            for z in range(mrc_dest.zdim):

                xyz_arr = np.array((x, y, z))

                # find the center in the old grid
                pos = (xyz_arr * mrc_dest.xwidth + mrc_dest.orig -
                       mrc_source.orig) / mrc_source.xwidth

                # calculate the index for 1d density vector representation
                ind = mrc_dest.xdim * mrc_dest.ydim * z + mrc_dest.xdim * y + x

                # skip calculation if position is outside of the old grid
                if (pos[0] < 0 or pos[1] < 0 or pos[2] < 0
                        or pos[0] >= mrc_source.xdim
                        or pos[1] >= mrc_source.ydim
                        or pos[2] >= mrc_source.zdim):
                    mrc_dest.dens[ind] = 0.0
                    mrc_dest.vec[x][y][z] = 0.0
                    continue

                # skip calculation if the old position has zero density
                if mrc_source.data[int(pos[0])][int(pos[1])][int(pos[2])] == 0:
                    mrc_dest.dens[ind] = 0.0
                    mrc_dest.vec[x][y][z] = 0.0
                    continue

                # Start Point
                stp = (pos - fmaxd).astype(np.int32)

                # End Point
                endp = (pos + fmaxd + 1).astype(np.int32)

                # initialize padding flags
                x_left_padding = False
                y_left_padding = False
                z_left_padding = False

                # set start and end point, add padding if applicable
                if stp[0] < 0:
                    x_left_padding = True
                    stp[0] = 0
                if stp[1] < 0:
                    y_left_padding = True
                    stp[1] = 0
                if stp[2] < 0:
                    z_left_padding = True
                    stp[2] = 0

                if endp[0] > mrc_source.xdim:
                    endp[0] = mrc_source.xdim
                if endp[1] > mrc_source.ydim:
                    endp[1] = mrc_source.ydim
                if endp[2] > mrc_source.zdim:
                    endp[2] = mrc_source.zdim
                # compute the total density
                selected_region = mrc_source.data[stp[0]:endp[0], stp[1]:endp[1],
                                  stp[2]:endp[2]]  # select the region to be sampled in the original map

                padding = kernel_length - (endp - stp)  # calculate padding values

                kernel_x_range, kernel_y_range, kernel_z_range = [0, 0], [0, 0], [0,
                                                                                  0]  # init padding values to be all zeros

                # apply the directions for padding values
                if padding[0] != 0:
                    if x_left_padding:
                        kernel_x_range = [padding[0], 0]
                    else:
                        kernel_x_range = [0, padding[0]]
                if padding[1] != 0:
                    if y_left_padding:
                        kernel_y_range = [padding[1], 0]
                    else:
                        kernel_y_range = [0, padding[1]]
                if padding[2] != 0:
                    if z_left_padding:
                        kernel_z_range = [padding[2], 0]
                    else:
                        kernel_z_range = [0, padding[2]]

                # apply padding to the selected region in old map
                padded_region = np.pad(selected_region, (
                    (kernel_x_range[0], kernel_x_range[1]), (kernel_y_range[0], kernel_y_range[1]),
                    (kernel_z_range[0], kernel_z_range[1])))

                # apply the kernel to the padded array
                dtotal, weighted_samples = apply_kern(padded_region, g_kern)

                # fill in the dens and data array
                mrc_dest.dens[ind] = dtotal
                mrc_dest.data[x][y][z] = dtotal

                # calculate the vector value using center_of_mass and normalize
                if calc_vec:
                    vec = np.array(ndimage.center_of_mass(weighted_samples) - np.array([kernel_length] * 3) / 2.)

                    if dtotal == 0:
                        mrc_dest.vec[x][y][z] = 0.0
                        continue

                    normalized_v = vec / np.sqrt(np.sum(vec ** 2))  # normalization to unit vector
                    mrc_dest.vec[x][y][z] = normalized_v  # fill in the vector array in new map

                dsum += dtotal
                Nact += 1

    print("#End LDP")
    print(dsum)
    print(Nact)

    mrc_dest.dsum = dsum
    mrc_dest.Nact = Nact
    mrc_dest.ave = dsum / float(Nact)
    mrc_dest.std = np.linalg.norm(mrc_dest.dens[mrc_dest.dens > 0])
    mrc_dest.std_norm_ave = np.linalg.norm(mrc_dest.dens[mrc_dest.dens > 0] -
                                           mrc_dest.ave)

    print("#MAP AVE={ave} STD={std} STD_norm={std_norm}".format(
        ave=mrc_dest.ave, std=mrc_dest.std, std_norm=mrc_dest.std_norm_ave))

    return mrc_dest


@numba.jit(nopython=True)
def rot_pos_mtx(mtx, vec):
    """Rotate a vector or matrix using a rotation matrix.

    Args:
        mtx (numpy.array): the rotation matrix
        vec (numpy.array): the vector/matrix to be rotated

    Returns:
        ret (numpy.array): the rotated vector/matrix
    """
    mtx = mtx.astype(np.float32)
    vec = vec.astype(np.float32)

    ret = vec @ mtx

    return ret


def rot_mrc(orig_mrc_data, orig_mrc_vec, angle):
    """A function to rotation the density and vector array by a specified angle.

    Args:
        orig_mrc_data (numpy.array): the data array to be rotated
        orig_mrc_vec (numpy.array): the vector array to be rotated
        angle (float, float, float): the angle of rotation in degrees

    Returns:
        new_vec_array (numpy.array): rotated vector array
        new_data_array (numpy.array): rotated data array
    """

    # set the dimension to be x dimension as all dimension are the same
    dim = orig_mrc_vec.shape[0]

    # create array for the positions after rotation
    new_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), )).T.reshape(-1, 3)

    # set the rotation center
    cent = 0.5 * float(dim)

    # get relative new positions from center
    new_pos = new_pos - cent

    # init the rotation matrix by euler angle
    r = R.from_euler("ZYX", angle, degrees=True)
    mtx = r.as_matrix()
    mtx[np.isclose(mtx, 0, atol=1e-15)] = 0

    # reversely rotate the new position lists to get old positions
    old_pos = rot_pos_mtx(np.flip(mtx).T, new_pos) + cent

    # concatenate combine two position array horizontally for later filtering
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

    # get the mask of all the values inside boundary
    combined_arr = combined_arr[in_bound_mask]

    # convert the index to integer
    combined_arr = combined_arr.astype(np.int32)

    # get the old index array
    index_arr = combined_arr[:, 0:3]

    # get the index that has non-zero density by masking
    dens_mask = orig_mrc_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    non_zero_rot_list = combined_arr[dens_mask]

    # get the non-zero vec and dens values
    non_zero_vec = orig_mrc_vec[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    non_zero_dens = orig_mrc_data[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    new_vec = rot_pos_mtx(np.flip(mtx), non_zero_vec)

    # init new vec and dens array
    new_vec_array = np.zeros_like(orig_mrc_vec)
    new_data_array = np.zeros_like(orig_mrc_data)

    # find the new indices
    new_ind_arr = (non_zero_rot_list[:, 3:6] + cent).astype(int)

    # fill in the values to new vec and dens array
    new_vec_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = new_vec
    new_data_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens

    return new_vec_array, new_data_array


def find_best_trans_list(input_list):
    """find the best translation based on list of FFT transformation results

    Args:
        input_list (numpy.array): FFT result list

    Returns:
        best (float): the maximum score found
        trans (list(int)): the best translation associated with the maximum score
    """

    sum_arr = np.zeros_like(input_list[0])
    for arr in input_list:
        sum_arr = sum_arr + arr
    best = np.amax(sum_arr)
    trans = np.unravel_index(sum_arr.argmax(), sum_arr.shape)

    return best, trans


def find_best_trans_list_prob(input_list, alpha):
    """find the best translation based on list of Z score normalised FFT transformation results

    Args:
        input_list (numpy.array): FFT result list, alpha: weighting parameter

    Returns:
        best (float): the maximum score found
        trans (list(int)): the best translation associated with the maximum score
        best_prob: the maximum probability normalised score
    """

    sum_arr = np.zeros_like(input_list[0])
    # for arr in input_list:
    dot_array = input_list[0] + input_list[1] + input_list[2]
    # avg_dot=np.sum(dot_array)/(dot_array.shape[0])
    ave_dot = np.mean(dot_array)
    std_dot = np.std(dot_array)
    dot_array_z = (dot_array - ave_dot) / std_dot

    prob_array = input_list[3] + input_list[4] + input_list[5] + input_list[6]
    # avg_dot=np.sum(dot_array)/(dot_array.shape[0])
    ave_prob = np.mean(prob_array)
    std_prob = np.std(prob_array)
    prob_array_z = (prob_array - ave_prob) / std_prob

    # sum_arr = sum_arr+input_list[0]+input_list[1]+input_list[2]+(input_list[3] + input_list[4] + input_list[5] + input_list[6])/5000000.00
    if alpha == 1:
        sum_arr = sum_arr + dot_array_z + prob_array_z
    else:
        sum_arr = sum_arr + (alpha) * dot_array_z + (1 - alpha) * prob_array_z
    prob_arr = input_list[3] + input_list[4] + input_list[5] + input_list[6]
    best = np.amax(sum_arr)
    best_prob = np.amax(prob_array_z)
    trans = np.unravel_index(sum_arr.argmax(), sum_arr.shape)

    return best, trans, best_prob


def fft_search_score_trans(target_X, target_Y, target_Z, search_vec, a, b, c, fft_object, ifft_object):
    """A function perform FFT transformation on the query density vectors and finds the best translation on 3D vectors.

    Args:
        target_X, target_Y, target_Z (numpy.array): FFT transformed result from target map for xyz axies
        search_vec (numpy.array): the input query map vector array
        a, b, c (numpy.array): empty n-bytes aligned arrays for holding intermediate values in the transformation
        fft_object (pyfftw.FFTW): preset FFT transformation plan
        ifft_object (pyfftw.FFTW): preset inverse FFT transformation plan

    Returns:
        best (float): the maximum score found
        trans (list(int)): the best translation associated with the maximum score
    """

    # make copies of the original vector arrays
    x2 = copy.deepcopy(search_vec[..., 0])
    y2 = copy.deepcopy(search_vec[..., 1])
    z2 = copy.deepcopy(search_vec[..., 2])

    # FFT tranformations and vector product
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

    return find_best_trans_list([dot_x, dot_y, dot_z])


def fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object):
    """A better version of the fft_search_score_trans function that finds the best dot product for the target and query list of vectors.

    Args:
        target_list (list(numpy.array)): FFT transformed result from target map (any dimensions)
        query_list (list(numpy.array)): the input query map vector array (must has the same dimensions as target_list)
        a, b, c (numpy.array): empty n-bytes aligned arrays for holding intermediate values in the transformation
        fft_object (pyfftw.FFTW): preset FFT transformation plan
        ifft_object (pyfftw.FFTW): preset inverse FFT transformation plan

    Returns:
        dot_product_list: (list(numpy.array)): vector product result that can be fed into find_best_trans_list() to find best translation
    """

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


@numba.jit(nopython=True)
def laplacian_filter(arr):
    """A simple laplacian filter applied to an array with the kernel [[0, 1, 0], [1, -6, 1], [0, 1, 0]].

    Args:
        arr (numpy.array): the array to be filtered

    Returns:
        new_arr (numpy.array): the filtered array
    """

    xdim = arr.shape[0]
    ydim = arr.shape[1]
    zdim = arr.shape[2]
    new_arr = np.zeros_like(arr)
    for x in range(xdim):
        for y in range(ydim):
            for z in range(zdim):
                if arr[x][y][z] > 0:
                    new_arr[x][y][z] = -6.0 * arr[x][y][z]
                    if (x + 1 < xdim):
                        new_arr[x][y][z] += arr[x + 1][y][z]
                    if (x - 1 >= 0):
                        new_arr[x][y][z] += arr[x - 1][y][z]
                    if (y + 1 < ydim):
                        new_arr[x][y][z] += arr[x][y + 1][z]
                    if (y - 1 >= 0):
                        new_arr[x][y][z] += arr[x][y - 1][z]
                    if (z + 1 < zdim):
                        new_arr[x][y][z] += arr[x][y][z + 1]
                    if (z - 1 >= 0):
                        new_arr[x][y][z] += arr[x][y][z - 1]
    return new_arr


def fft_search_score_trans_1d(target_X, search_data, a, b, fft_object, ifft_object, mode, ave=None):
    """1D version of fft_search_score_trans.

    Args:
        target_X (numpy.array): FFT transformed result from target map in 1D
        search_data (numpy.array): the input query map array in 1D
        a, b (numpy.array): empty n-bytes aligned arrays for holding intermediate values in the transformation
        fft_object (pyfftw.FFTW): preset FFT transformation plan
        ifft_object (pyfftw.FFTW): preset inverse FFT transformation plan
        mode (string): special mode to use: Overlap, CC, PCC, Laplacian
        ave (float, optional): placeholder for average value. Defaults to None.

    Returns:
        best (float): the maximum score found
        trans (list(int)): the best translation associated with the maximum score
    """

    x2 = copy.deepcopy(search_data)

    if mode == "Overlap":
        x2 = np.where(x2 > 0, 1.0, 0.0)
    elif mode == "CC":
        x2 = np.where(x2 > 0, x2, 0.0)
    elif mode == "PCC":
        x2 = np.where(x2 > 0, x2 - ave, 0.0)
    elif mode == "Laplacian":
        x2 = laplacian_filter(x2)

    X2 = np.zeros_like(target_X)
    np.copyto(a, x2)
    np.copyto(X2, fft_object(a))
    dot_X = target_X * X2
    np.copyto(b, dot_X)
    dot_x = np.zeros_like(x2)
    np.copyto(dot_x, ifft_object(b))

    return find_best_trans_list([dot_x])


def search_map_fft(mrc_target, mrc_search, TopN=10, ang=30, mode="VecProduct", is_eval_mode=False, save_path="."):
    """The main search function for fining the best superimposition for the target and the query map.

    Args:
        mrc_target (mrc_obj): the input target map
        mrc_search (mrc_obj): the input query map
        TopN (int, optional): the number of top superimposition to find. Defaults to 10.
        ang (int, optional): search interval for angular rotation. Defaults to 30.
        mode (str, optional): special modes to use. Defaults to "VecProduct".
        is_eval_mode (bool, optional): set the evaluation mode true will only perform scoring but not searching. Defaults to False.
        save_path (str, optional): the path to save output .pdb files. Defaults to the current directory.

    Returns:
        refined_list (list): a list of refined search results
    """

    time_start = time.time()

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

        _ = get_score(mrc_target.data,
                      mrc_search.data,
                      mrc_target.vec,
                      mrc_search.vec,
                      [0, 0, 0],
                      mrc_target.ave,
                      mrc_search.ave,
                      mrc_target.std,
                      mrc_search.std,
                      mrc_target.std_norm_ave,
                      mrc_search.std_norm_ave)
        return None

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
        x1 = laplacian_filter(mrc_target.data)

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

    # rot_vec_dict, rot_data_dict = rot_init_cuda(mrc_search.data, mrc_search.vec, angle_comb)

    rot_vec_dict = {}
    rot_data_dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() + 4) as executor:
        trans_vec = {executor.submit(rot_mrc, mrc_search.data, mrc_search.vec, angle, ): angle for angle in angle_comb}
        for future in concurrent.futures.as_completed(trans_vec):
            angle = trans_vec[future]
            rot_vec_dict[tuple(angle)] = future.result()[0]
            rot_data_dict[tuple(angle)] = future.result()[1]

    time_rot = time.time()

    print("Rotation time: " + str(time_rot - time_start))

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

    time_fft = time.time()

    print("FFT time: " + str(time_fft - time_rot))

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
            trans_vec = {executor.submit(rot_mrc, mrc_search.data, mrc_search.vec, angle, ): angle for angle in
                         refine_ang_arr}
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

        # sort the list to find the TopN with best scores
        refined_list = sorted(refined_score, key=lambda x: x[1], reverse=True)[:TopN]

    else:
        # no action taken when refinement is disabled
        refined_list = sorted_topN

    # calculate the refinement time
    time_refine = time.time()
    print("Refinement time: " + str(time_refine - time_fft))

    # Save the results to file
    for i, t_mrc in enumerate(refined_list):
        # calculate the scores
        print()
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
        )

        # Write result to PDB files
        show_vec(mrc_target.orig, t_mrc[3], t_mrc[4], sco, mrc_search.xwidth, t_mrc[2],
                 "model_top_" + str(i + 1) + ".pdb", save_path)

    time_writefile = time.time()

    print("File Write time: " + str(time_writefile - time_refine))

    return refined_list


def search_map_fft_prob(mrc_P1, mrc_P2, mrc_P3, mrc_P4, mrc_target, mrc_search, mrc_search_p1, mrc_search_p2,
                        mrc_search_p3, mrc_search_p4, alpha=1, TopN=10, ang=10, mode="VecProduct", is_eval_mode=False):
    """The main search function for fining the best superimposition for the target and the query map.

    Args:
        mrc_P1: large probability map1
        mrc_P2: large probability map2
        mrc_P3: large probability map3
        mrc_P4: large probability map4
        mrc_search_p1: query probability map1
        mrc_search_p2: query probability map2
        mrc_search_p3: query probability map3
        mrc_search_p4: query probability map4
        alpha: weighting parameter
        mrc_target (mrc_obj): the input target map
        mrc_search (mrc_obj): the input query map
        TopN (int, optional): the number of top superimposition to find. Defaults to 10.
        ang (int, optional): search interval for angular rotation. Defaults to 30.
        mode (str, optional): special modes to use. Defaults to "VecProduct".
        is_eval_mode (bool, optional): set the evaluation mode true will only perform scoring but not searching. Defaults to False.
        save_path (str, optional): the path to save output .pdb files. Defaults to the current directory.

    Returns:
        refined_list (list): a list of refined search results including the probability score
    """

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

    # init the target map vectors
    # does this part need to be changed?
    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    p1 = copy.deepcopy(mrc_P1.data)
    p2 = copy.deepcopy(mrc_P2.data)
    p3 = copy.deepcopy(mrc_P3.data)
    p4 = copy.deepcopy(mrc_P4.data)

    y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
    z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    # Score normalization constant

    d3 = mrc_target.xdim ** 3

    rd3 = 1.0 / d3

    # Calculate the FFT results for target map

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

    Y1 = np.fft.rfftn(y1)
    Y1 = np.conj(Y1)
    Z1 = np.fft.rfftn(z1)
    Z1 = np.conj(Z1)

    # Compose target result list

    target_list = [X1, Y1, Z1, P1, P2, P3, P4]

    # Calculate all the combination of angles

    angle_comb = calc_angle_comb(ang)

    # Paralleled processing for rotation and FFT process per angle in angle_comb

    angle_score = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        rets = {
            executor.submit(rot_and_search_fft,
                            mrc_search.data,
                            mrc_search.vec,
                            mrc_search_p1.data,
                            mrc_search_p2.data,
                            mrc_search_p3.data,
                            mrc_search_p4.data,
                            angle,
                            target_list,
                            alpha): angle for angle in angle_comb}
        for future in tqdm(concurrent.futures.as_completed(rets), total=len(angle_comb)):
            angle = rets[future]
            angle_score.append([tuple(angle), future.result()[0] * rd3, future.result()[1], future.result()[2]])

    # calculate the ave and std for all the rotations
    score_arr = np.array([row[1] for row in angle_score])
    ave = np.mean(score_arr)
    std = np.std(score_arr)
    print("Std= " + str(std) + " Ave= " + str(ave))

    # sort the list and get topN
    sorted_topN = sorted(angle_score, key=lambda x: x[1], reverse=True)[:TopN]

    # print TopN Statistics
    for x in sorted_topN:
        print(x)

    refined_score = []
    if ang > 5.0:
        # Search for +5.0 and -5.0 degree rotation.

        for t_mrc in sorted_topN:
            ang_list = np.array(
                np.meshgrid(
                    [t_mrc[0][0] - 5, t_mrc[0][0], t_mrc[0][0] + 5],
                    [t_mrc[0][1] - 5, t_mrc[0][1], t_mrc[0][1] + 5],
                    [t_mrc[0][2] - 5, t_mrc[0][2], t_mrc[0][2] + 5],
                )
            ).T.reshape(-1, 3)

            for ang in ang_list:
                rotated = rot_mrc_prob(mrc_search.data, mrc_search.vec, mrc_search_p1.data, mrc_search_p2.data,
                                       mrc_search_p3.data, mrc_search_p4.data, ang)
                rotated_vec = rotated[0]
                rotated_data = rotated[1]

                x2 = copy.deepcopy(rotated_vec[..., 0])
                y2 = copy.deepcopy(rotated_vec[..., 1])
                z2 = copy.deepcopy(rotated_vec[..., 2])
                p21 = copy.deepcopy(rotated[2])
                p22 = copy.deepcopy(rotated[3])
                p23 = copy.deepcopy(rotated[4])
                p24 = copy.deepcopy(rotated[5])

                target_list = [X1, Y1, Z1, P1, P2, P3, P4]
                query_list = [x2, y2, z2, p21, p22, p23, p24]

                # fftw plans
                a = pyfftw.empty_aligned((x2.shape), dtype="float32")
                b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
                c = pyfftw.empty_aligned((x2.shape), dtype="float32")

                fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
                ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

                fft_result_list = fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object)

                best, trans, best_prob = find_best_trans_list_prob(fft_result_list, alpha)

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

    return refined_list


def show_vec(origin, sampled_mrc_vec, sampled_mrc_data, sampled_mrc_score, sample_width, trans, name):
    dim = sampled_mrc_data.shape[0]

    origin = np.array([origin[0], origin[1], origin[2]])
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
def rot_mrc_prob(orig_mrc_data, orig_mrc_vec, mrc_search_p1_data, mrc_search_p2_data, mrc_search_p3_data,
                 mrc_search_p4_data, angle):
    dim = orig_mrc_vec.shape[0]

    new_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), )).T.reshape(-1, 3)

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

    non_zero_dens_p1 = mrc_search_p1_data[
        non_zero_rot_list_p1[:, 0], non_zero_rot_list_p1[:, 1], non_zero_rot_list_p1[:, 2]]

    non_zero_dens_p2 = mrc_search_p2_data[
        non_zero_rot_list_p2[:, 0], non_zero_rot_list_p2[:, 1], non_zero_rot_list_p2[:, 2]]

    non_zero_dens_p3 = mrc_search_p3_data[
        non_zero_rot_list_p3[:, 0], non_zero_rot_list_p3[:, 1], non_zero_rot_list_p3[:, 2]]

    non_zero_dens_p4 = mrc_search_p4_data[
        non_zero_rot_list_p4[:, 0], non_zero_rot_list_p4[:, 1], non_zero_rot_list_p4[:, 2]]

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

    target_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), )).T.reshape(-1, 3)

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
        target_map_data, search_map_data, target_map_vec, search_map_vec, trans, ave1, ave2, std1, std2, pstd1, pstd2,
        sco_prob_added, sco_prob_only
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

    # duplicate this part for all probability maps

    target_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), )).T.reshape(-1, 3)

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


def calc_angle_comb(ang_interval):
    """ Calculate the all the possible combination of angles given the interval in degrees"""

    x_angle = []
    y_angle = []
    z_angle = []

    i = 0
    while i < 360:
        x_angle.append(i)
        y_angle.append(i)
        i += ang_interval

    i = 0
    while i <= 180:
        z_angle.append(i)
        i += ang_interval

    angle_comb = np.array(np.meshgrid(x_angle, y_angle, z_angle)).T.reshape(-1, 3)
    return angle_comb


def rot_and_search_fft(data, vec, dp1, dp2, dp3, dp4, angle, target_list, alpha):
    """ Calculate the best translation for the query map given a rotation angle

    Args:
        data (numpy.array): The data of query map
        vec (numpy.array): The vector representation of query map
        dp1-dp4 (numpy.array): The probability representation of query map
        angle (list/tuple): The rotation angle
        target_list (numpy.array) : A list of FFT-transformed results of the target map
        alpha (float): Parameter for alpha mixing during dot score calculation

    Returns:
        best (float): Best score calculated using FFT
        trans (list): Best translation in [x,y,z]
    """

    # Rotate the query map and vector representation
    new_vec_array, new_data_array, new_data_array_p1, new_data_array_p2, new_data_array_p3, new_data_array_p4 = rot_mrc_prob(
        data, vec, dp1, dp2, dp3, dp4, angle)

    # Compose the query FFT list

    x2 = copy.deepcopy(new_vec_array[..., 0])
    y2 = copy.deepcopy(new_vec_array[..., 1])
    z2 = copy.deepcopy(new_vec_array[..., 2])
    p21 = copy.deepcopy(new_data_array_p1)
    p22 = copy.deepcopy(new_data_array_p2)
    p23 = copy.deepcopy(new_data_array_p3)
    p24 = copy.deepcopy(new_data_array_p4)

    query_list = [x2, y2, z2, p21, p22, p23, p24]

    # fftw plans initialization
    a = pyfftw.empty_aligned((x2.shape), dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned((x2.shape), dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    # Search for best translation using FFT
    fft_result_list = fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object)

    best, trans, best_prob = find_best_trans_list_prob(fft_result_list, alpha)

    return best, trans, best_prob
