# coding: utf-8
# import concurrent.futures
import copy
import mrcfile
import multiprocessing
import numba
import numpy as np
import pyfftw
from datetime import datetime
from pathlib import Path
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

    print()
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
                v = mrc1_data[xp][yp][zp] * np.exp(-1.5 * d2 * fsiv)
                dtotal += v
                pos2[0] += v * xp
                pos2[1] += v * yp
                pos2[2] += v * zp

    return dtotal, pos2


def fastVEC(src, dest, dreso=16.0):
    src_xwidth = src.xwidth
    src_orig = src.orig
    src_dims = np.array((src.xdim, src.ydim, src.zdim))
    dest_xwidth = dest.xwidth
    dest_orig = dest.orig
    dest_dims = np.array((dest.xdim, dest.ydim, dest.zdim))

    dest_data, dest_vec = doFastVEC(src_xwidth, src_orig, src_dims, src.data,
                                    dest_xwidth, dest_orig, dest_dims, dest.data, dest.vec,
                                    dreso)

    dsum = np.sum(dest_data)
    Nact = np.count_nonzero(dest_data)
    ave = np.mean(dest_data[dest_data > 0])
    std = np.linalg.norm(dest_data[dest_data > 0])
    std_norm_ave = np.linalg.norm(dest_data[dest_data > 0] - ave)

    print("#MAP SUM={sum} COUNT={cnt} AVE={ave} STD={std} STD_norm={std_norm}".format(sum=dsum,
                                                                                      cnt=Nact,
                                                                                      ave=ave,
                                                                                      std=std,
                                                                                      std_norm=std_norm_ave))

    dest.data = dest_data
    dest.vec = dest_vec
    dest.dsum = dsum
    dest.Nact = Nact
    dest.ave = ave
    dest.std = std
    dest.std_norm_ave = std_norm_ave

    return dest


@numba.jit(nopython=True)
def doFastVEC(src_xwidth, src_orig, src_dims, src_data, dest_xwidth, dest_orig, dest_dims, dest_data, dest_vec,
              dreso=16.0):
    gstep = src_xwidth
    fs = (dreso / gstep) * 0.5
    fs = fs ** 2
    fsiv = 1.0 / fs
    fmaxd = (dreso / gstep) * 2.0

    # print("#maxd={fmaxd}".format(fmaxd=fmaxd), "#fsiv=" + str(fsiv))

    for x in range(dest_dims[0]):
        for y in range(dest_dims[1]):
            for z in range(dest_dims[2]):

                xyz_arr = np.array((x, y, z))
                pos = (xyz_arr * dest_xwidth + dest_orig - src_orig) / src_xwidth

                # check density

                if (
                        pos[0] < 0
                        or pos[1] < 0
                        or pos[2] < 0
                        or pos[0] >= src_dims[0]
                        or pos[1] >= src_dims[1]
                        or pos[2] >= src_dims[2]
                ):
                    continue

                if src_data[int(pos[0])][int(pos[1])][int(pos[2])] == 0:
                    continue

                # Start Point
                stp = (pos - fmaxd).astype(np.int32)

                # set start and end point
                if stp[0] < 0:
                    stp[0] = 0
                if stp[1] < 0:
                    stp[1] = 0
                if stp[2] < 0:
                    stp[2] = 0

                # End Point
                endp = (pos + fmaxd + 1).astype(np.int32)

                if endp[0] >= src_dims[0]:
                    endp[0] = src_dims[0]
                if endp[1] >= src_dims[1]:
                    endp[1] = src_dims[1]
                if endp[2] >= src_dims[2]:
                    endp[2] = src_dims[2]

                # compute the total density
                dtotal, pos2 = calc(stp, endp, pos, src_data, fsiv)

                dest_data[x][y][z] = dtotal

                if dtotal == 0:
                    continue

                rd = 1.0 / dtotal

                pos2 *= rd

                tmpcd = pos2 - pos

                dvec = np.sqrt(tmpcd[0] ** 2 + tmpcd[1] ** 2 + tmpcd[2] ** 2)

                if dvec == 0:
                    dvec = 1.0

                rdvec = 1.0 / dvec

                dest_vec[x][y][z] = tmpcd * rdvec

    return dest_data, dest_vec


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

    # FFT transformations and vector product
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
    """A better version of the fft_search_score_trans function that finds the best dot product for the target and
    query list of vectors.

    Args:
        target_list (list(numpy.array)): FFT transformed result from target map (any dimensions)
        query_list (list(numpy.array)): the input query map vector array (must has the same dimensions as target_list)
        a, b, c (numpy.array): empty n-bytes aligned arrays for holding intermediate values in the transformation
        fft_object (pyfftw.FFTW): preset FFT transformation plan
        ifft_object (pyfftw.FFTW): preset inverse FFT transformation plan

    Returns: dot_product_list: (list(numpy.array)): vector product result that can be fed into find_best_trans_list()
    to find best translation
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


def search_map_fft(mrc_target, mrc_search, TopN=10, ang=30, mode="VecProduct", is_eval_mode=False, save_path=".",
                   showPDB=False, folder=None):
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

        _ = get_score(mrc_target, mrc_search.data, mrc_search.vec, [0, 0, 0])

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

    angle_comb = calc_angle_comb(ang)

    # init the target map vectors lists

    if mode == "VecProduct":
        target_list = [X1, Y1, Z1]
    else:
        target_list = [X1]

    print("###Start Searching###")

    angle_score = []

    # search process
    for angle in tqdm(angle_comb):
        vec_score, vec_trans, _, _ = rot_and_search_fft(mrc_search.data,
                                                        mrc_search.vec,
                                                        angle,
                                                        target_list,
                                                        mrc_target,
                                                        mode=mode)
        angle_score.append({
            "angle": angle,
            "vec_score": vec_score * rd3,
            "vec_trans": vec_trans
        })

    # calculate the ave and std
    score_arr = np.array([row["vec_score"] for row in angle_score])
    ave = np.mean(score_arr)
    std = np.std(score_arr)
    print("\nStd= " + str(std) + " Ave= " + str(ave) + "\n")

    # sort the list and get topN
    sorted_topN = sorted(angle_score, key=lambda x: x["vec_score"], reverse=True)[:TopN]

    for i, result_mrc in enumerate(sorted_topN):
        r = euler2rot(result_mrc["angle"])
        new_trans = convert_trans(mrc_target.cent,
                                  mrc_search.cent,
                                  r,
                                  result_mrc["vec_trans"],
                                  mrc_search.xwidth,
                                  mrc_search.xdim)

        print("M" + str(i),
              "Rotation=",
              "(" + str(result_mrc["angle"][0]),
              str(result_mrc["angle"][1]),
              str(result_mrc["angle"][2]) + ")",
              "Translation=",
              "(" + "{:.3f}".format(new_trans[0]),
              "{:.3f}".format(new_trans[1]),
              "{:.3f}".format(new_trans[2]) + ")"
              )

    print()

    refined_score = []
    if ang > 5.0:

        # setup all the angles for refinement
        # initialize the refinement list by ±5 degrees
        refine_ang_list = []
        for result_mrc in sorted_topN:
            ang = result_mrc["angle"]
            ang_list = np.array(
                np.meshgrid(
                    [ang[0] - 5, ang[0], ang[0] + 5],
                    [ang[1] - 5, ang[1], ang[1] + 5],
                    [ang[2] - 5, ang[2], ang[2] + 5],
                )
            ).T.reshape(-1, 3)

            # sanity check
            ang_list = ang_list[(ang_list[:, 0] < 360) &
                                (ang_list[:, 1] < 360) &
                                (ang_list[:, 2] < 180)]

            ang_list[ang_list < 0] += 360

            refine_ang_list.append(ang_list)

        refine_ang_arr = np.concatenate(refine_ang_list, axis=0)

        # rotate the mrc vector and data according to the list (multi-threaded)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() + 4) as executor:
        #     trans_vec = {executor.submit(rot_mrc, mrc_search.data, mrc_search.vec, angle, ): angle for angle in
        #                  refine_ang_arr}
        #     for future in concurrent.futures.as_completed(trans_vec):
        #         angle = trans_vec[future]
        #         rot_vec_dict[tuple(angle)] = future.result()[0]
        #         rot_data_dict[tuple(angle)] = future.result()[1]

        for angle in tqdm(refine_ang_arr, desc="Refining Rotation"):
            vec_score, vec_trans, new_vec, new_data = rot_and_search_fft(mrc_search.data,
                                                                         mrc_search.vec,
                                                                         angle,
                                                                         target_list,
                                                                         mrc_target,
                                                                         mode=mode)

            refined_score.append({"angle": tuple(angle),
                                  "vec_score": vec_score * rd3,
                                  "vec_trans": vec_trans,
                                  "vec": new_vec,
                                  "data": new_data})

        # sort the list to find the TopN with best scores
        refined_list = sorted(refined_score, key=lambda x: x["vec_score"], reverse=True)[:TopN]

    else:
        # no action taken when refinement is disabled
        refined_list = sorted_topN

    # Write result to PDB files
    if showPDB:
        if folder is not None:
            folder_path = Path.cwd() / folder
        else:
            folder_path = Path.cwd() / ("VESPER_RUN_" + datetime.now().strftime('%m%d_%H%M'))
        Path.mkdir(folder_path)
        print("\n###Writing results to PDB files###")
    else:
        print()

    for i, result_mrc in enumerate(refined_list):
        r = euler2rot(result_mrc["angle"])
        new_trans = convert_trans(mrc_target.cent,
                                  mrc_search.cent,
                                  r,
                                  result_mrc["vec_trans"],
                                  mrc_search.xwidth,
                                  mrc_search.xdim)

        print("\n#" + str(i),
              "Rotation=",
              "(" + str(result_mrc["angle"][0]),
              str(result_mrc["angle"][1]),
              str(result_mrc["angle"][2]) + ")",
              "Translation=",
              "(" + "{:.3f}".format(new_trans[0]),
              "{:.3f}".format(new_trans[1]),
              "{:.3f}".format(new_trans[2]) + ")"
              )

        sco_arr = get_score(
            mrc_target,
            result_mrc["data"],
            result_mrc["vec"],
            result_mrc["vec_trans"]
        )

        if showPDB:
            show_vec(mrc_target.orig,
                     result_mrc["vec"],
                     result_mrc["data"],
                     sco_arr,
                     result_mrc["vec_score"],
                     mrc_search.xwidth,
                     result_mrc["vec_trans"],
                     result_mrc["angle"],
                     folder_path,
                     i)

    return refined_list


def search_map_fft_prob(mrc_P1, mrc_P2, mrc_P3, mrc_P4,
                        mrc_target, mrc_search,
                        mrc_search_p1, mrc_search_p2, mrc_search_p3, mrc_search_p4,
                        ang, alpha=0.0, TopN=10, num_proc=4,
                        vave=-10, vstd=-10, pave=-10, pstd=-10,
                        showPDB=False,
                        folder=None):
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

    # init the target map vectors
    # does this part need to be changed?
    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
    z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    p1 = copy.deepcopy(mrc_P1.data)
    p2 = copy.deepcopy(mrc_P2.data)
    p3 = copy.deepcopy(mrc_P3.data)
    p4 = copy.deepcopy(mrc_P4.data)

    # Score normalization constant

    rd3 = 1.0 / (mrc_target.xdim ** 3)

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

    print()
    print("###Start Searching###")

    angle_score = []

    if vave >= 0 and vstd >= 0 and pstd >= 0 and pave >= 0:
        pass
    else:
        for angle in tqdm(angle_comb):
            vec_score, vec_trans, prob_score, prob_trans, _, _ = rot_and_search_fft_prob(mrc_search.data,
                                                                                         mrc_search.vec,
                                                                                         mrc_search_p1.data,
                                                                                         mrc_search_p2.data,
                                                                                         mrc_search_p3.data,
                                                                                         mrc_search_p4.data,
                                                                                         angle,
                                                                                         target_list,
                                                                                         alpha=0.0)
            angle_score.append({
                "angle": angle,
                "vec_score": vec_score * rd3,
                "vec_trans": vec_trans,
                "prob_score": prob_score * rd3,
                "prob_trans": prob_trans
            })

        # multiprocessing version
        # with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
        #     rets = {
        #         executor.submit(rot_and_search_fft_prob,
        #                         mrc_search.data,
        #                         mrc_search.vec,
        #                         mrc_search_p1.data,
        #                         mrc_search_p2.data,
        #                         mrc_search_p3.data,
        #                         mrc_search_p4.data,
        #                         angle,
        #                         target_list,
        #                         alpha=0.0): angle for angle in angle_comb}
        #     for future in tqdm(concurrent.futures.as_completed(rets), total=len(angle_comb)):
        #         angle = rets[future]
        #         angle_score.append([tuple(angle),
        #                             future.result()[0] * rd3,
        #                             future.result()[1],
        #                             future.result()[2] * rd3,
        #                             future.result()[3]])

        # calculate the ave and std for all the rotations
        score_arr_vec = np.array([row["vec_score"] for row in angle_score])
        score_arr_prob = np.array([row["prob_score"] for row in angle_score])

        vstd = np.std(score_arr_vec)
        vave = np.mean(score_arr_vec)
        pstd = np.std(score_arr_prob)
        pave = np.mean(score_arr_prob)

    print()
    print("DotScore Std=", vstd, "DotScore Ave=", vave)
    print("ProbScore Std=", pstd, "ProbScore Ave=", pave)
    print()

    angle_score = []

    for angle in tqdm(angle_comb):
        vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans = rot_and_search_fft_prob(
            mrc_search.data,
            mrc_search.vec,
            mrc_search_p1.data,
            mrc_search_p2.data,
            mrc_search_p3.data,
            mrc_search_p4.data,
            angle,
            target_list,
            alpha,
            vstd, vave, pstd,
            pave)

        if mixed_score is None:
            mixed_score = 0
        if mixed_trans is None:
            mixed_trans = []

        angle_score.append({
            "angle": angle,
            "vec_score": vec_score * rd3,
            "vec_trans": vec_trans,
            "prob_score": prob_score * rd3,
            "prob_trans": prob_trans,
            "mixed_score": mixed_score * rd3,
            "mixed_trans": mixed_trans
        })

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_proc) as executor:
    #     rets = {
    #         executor.submit(rot_and_search_fft,
    #                         mrc_search.data,
    #                         mrc_search.vec,
    #                         mrc_search_p1.data,
    #                         mrc_search_p2.data,
    #                         mrc_search_p3.data,
    #                         mrc_search_p4.data,
    #                         angle,
    #                         target_list,
    #                         alpha,
    #                         vstd,
    #                         vave,
    #                         pstd,
    #                         pave): angle for angle in angle_comb}
    #
    #     for future in tqdm(concurrent.futures.as_completed(rets), total=len(angle_comb)):
    #         angle = rets[future]
    #         angle_score.append({"angle": tuple(angle),
    #                             "vec_score": future.result()[0] * rd3,
    #                             "vec_trans": future.result()[1],
    #                             "prob_score": future.result()[2] * rd3,
    #                             "prob_trans": future.result()[3],
    #                             "mixed_score": future.result()[4] * rd3,
    #                             "mixed_trans": future.result()[5]})

    # sort the list and get topN
    sorted_topN = sorted(angle_score, key=lambda x: x["mixed_score"], reverse=True)[:TopN]

    # print TopN Statistics
    for idx, x in enumerate(sorted_topN):
        print("M", str(idx + 1), x)

    refined_score = []
    if ang > 5.0:
        # Search for +5.0 and -5.0 degree rotation.
        print("\n###Start Refining###")

        refine_ang_list = []
        for result_mrc in sorted_topN:
            ang = result_mrc["angle"]
            ang_list = np.array(
                np.meshgrid(
                    [ang[0] - 5, ang[0], ang[0] + 5],
                    [ang[1] - 5, ang[1], ang[1] + 5],
                    [ang[2] - 5, ang[2], ang[2] + 5],
                )
            ).T.reshape(-1, 3)

            # remove duplicates
            ang_list = ang_list[(ang_list[:, 0] < 360) &
                                (ang_list[:, 1] < 360) &
                                (ang_list[:, 2] < 180)]

            ang_list[ang_list < 0] += 360

            refine_ang_list.append(ang_list)

        refine_ang_list = np.concatenate(refine_ang_list, axis=0)

        for ang in refine_ang_list:
            rotated = rot_mrc_prob(mrc_search.data, mrc_search.vec, mrc_search_p1.data, mrc_search_p2.data,
                                   mrc_search_p3.data, mrc_search_p4.data, ang)

            rotated_vec = rotated[0]
            rotated_data = rotated[1]

            x2 = rotated_vec[..., 0]
            y2 = rotated_vec[..., 1]
            z2 = rotated_vec[..., 2]

            p21 = rotated[2]
            p22 = rotated[3]
            p23 = rotated[4]
            p24 = rotated[5]

            target_list = [X1, Y1, Z1, P1, P2, P3, P4]
            query_list_vec = [x2, y2, z2]
            query_list_prob = [p21, p22, p23, p24]

            # fftw plans
            a = pyfftw.empty_aligned((x2.shape), dtype="float32")
            b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
            c = pyfftw.empty_aligned((x2.shape), dtype="float32")

            fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
            ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

            # Search for best translation using FFT
            fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec, a, b, c, fft_object,
                                                      ifft_object)
            fft_result_list_prob = fft_search_best_dot(target_list[3:], query_list_prob, a, b, c, fft_object,
                                                       ifft_object)

            vec_score, vec_trans = find_best_trans_list(fft_result_list_vec)
            prob_score, prob_trans = find_best_trans_list(fft_result_list_prob)

            mixed_score = 0
            mixed_trans = []

            mixed_score, mixed_trans = find_best_trans_mixed(fft_result_list_vec,
                                                             fft_result_list_prob,
                                                             alpha,
                                                             vstd, vave,
                                                             pstd, pave)

            refined_score.append(
                {"angle": tuple(ang),
                 "vec_score": vec_score * rd3,
                 "vec_trans": vec_trans,
                 "vec": rotated_vec,
                 "data": rotated_data,
                 "prob_score": prob_score * rd3,
                 "prob_trans": prob_trans,
                 "mixed_score": mixed_score * rd3,
                 "mixed_trans": mixed_trans})

        refined_list = sorted(refined_score, key=lambda x: x["mixed_score"], reverse=True)[:TopN]  # Sort by mixed score
    else:
        refined_list = sorted_topN

    if showPDB:
        # Write result to PDB files
        if folder is not None:
            folder_path = Path.cwd() / folder
        else:
            folder_path = Path.cwd() / ("VESPER_RUN_" + datetime.now().strftime('%m%d_%H%M'))
        Path.mkdir(folder_path)
        print("###Writing results to PDB files###")
    else:
        print()

    for i, result_mrc in enumerate(refined_list):
        # convert the translation back to the original coordinate system
        r = euler2rot(result_mrc["angle"])
        new_trans = convert_trans(mrc_target.cent,
                                  mrc_search.cent,
                                  r,
                                  result_mrc["mixed_trans"],
                                  mrc_search.xwidth,
                                  mrc_search.xdim)

        print("\n#" + str(i),
              "Rotation=",
              "(" + str(result_mrc["angle"][0]),
              str(result_mrc["angle"][1]),
              str(result_mrc["angle"][2]) + ")",
              "Translation=",
              "(" + "{:.3f}".format(new_trans[0]),
              "{:.3f}".format(new_trans[1]),
              "{:.3f}".format(new_trans[2]) + ")"
              )

        print("Translation=", str(result_mrc["vec_trans"]),
              "Probability Score=", str(result_mrc["prob_score"]),
              "Probability Translation=", str(result_mrc["prob_trans"]))

        sco_arr = get_score(
            mrc_target,
            result_mrc["data"],
            result_mrc["vec"],
            result_mrc["vec_trans"]
        )

        if showPDB:
            show_vec(mrc_target.orig,
                     result_mrc["vec"],
                     result_mrc["data"],
                     sco_arr,
                     result_mrc["mixed_score"],
                     mrc_search.xwidth,
                     result_mrc["mixed_trans"],
                     result_mrc["angle"],
                     folder_path,
                     i)

    return refined_list


def show_vec(origin,
             sampled_mrc_vec,
             sampled_mrc_data,
             score_arr,
             score,
             sample_width,
             trans,
             angle,
             folder_path,
             rank):
    dim = sampled_mrc_data.shape[0]

    filename = "R_{:02d}-S_{:>7.3f}.pdb".format(rank, score).replace(" ", "_")
    # filename = "M_{:02d}-S_{:>7.3f}-A_{:>5.1f}_{:>5.1f}_{:>5.1f}-T_{:>3.0f}_{:>3.0f}_{:>3.0f}.pdb".format(
    #     rank,
    #     score,
    #     angle[0],
    #     angle[1],
    #     angle[2],
    #     trans[0],
    #     trans[1],
    #     trans[2]).replace(" ", "_")

    filepath = folder_path / filename

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

    pdb_file = open(filepath, "w")
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):

                if sampled_mrc_data[x][y][z] != 0.0:
                    tmp = np.array([x, y, z])
                    tmp = tmp * sample_width + add
                    atom_header = "ATOM{:>7d}  CA  ALA{:>6d}    ".format(natm, nres)
                    atom_content = "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format(
                        tmp[0], tmp[1], tmp[2], 1.0, score_arr[x][y][z]
                    )
                    pdb_file.write(atom_header + atom_content + "\n")
                    natm += 1

                    tmp = np.array([x, y, z])
                    tmp = (tmp + sampled_mrc_vec[x][y][z]) * sample_width + add
                    atom_header = "ATOM{:>7d}  CB  ALA{:>6d}    ".format(natm, nres)
                    atom_content = "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}".format(
                        tmp[0], tmp[1], tmp[2], 1.0, score_arr[x][y][z]
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


def get_score(
        target_map,
        search_map_data,
        search_map_vec,
        trans
):
    target_map_data = target_map.data
    target_map_vec = target_map.vec

    ave1 = target_map.ave
    ave2 = np.mean(search_map_data[search_map_data > 0])

    std1 = target_map.std
    std2 = np.linalg.norm(search_map_data[search_map_data > 0])

    pstd1 = target_map.std_norm_ave
    pstd2 = np.linalg.norm(search_map_data[search_map_data > 0] - ave2)

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
        "Overlap="
        + str(float(Nm) / float(total))
        + " "
        + str(Nm) + "/" + str(total)
        + " CC="
        + str(cc / (std1 * std2))
        + " PCC="
        + str(pcc / (pstd1 * pstd2))
    )

    print("Score=", sco_sum)
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


def rot_and_search_fft(data, vec, angle, target_list, mrc_target, mode="VecProduct"):
    # Rotate the query map and vector representation
    new_vec, new_data = rot_mrc(data, vec, angle)

    # fftw plans initialization
    a = pyfftw.empty_aligned(new_vec[..., 0].shape, dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned(new_vec[..., 0].shape, dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    if mode == "VecProduct":

        # Compose the query FFT list

        x2 = new_vec[..., 0]
        y2 = new_vec[..., 1]
        z2 = new_vec[..., 2]

        query_list_vec = [x2, y2, z2]

        # Search for best translation using FFT
        fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec, a, b, c, fft_object, ifft_object)

        vec_score, vec_trans = find_best_trans_list(fft_result_list_vec)

    else:

        vec_score, vec_trans = fft_search_score_trans_1d(target_list[0],
                                                         new_data, a, b,
                                                         fft_object, ifft_object, mode,
                                                         mrc_target.ave)

        if mode == "CC":
            rstd2 = 1.0 / mrc_target.std ** 2
            vec_score = vec_score * rstd2
        if mode == "PCC":
            rstd3 = 1.0 / mrc_target.std_norm_ave ** 2
            vec_score = vec_score * rstd3

    return vec_score, vec_trans, new_vec, new_data


def rot_and_search_fft_prob(data, vec,
                            dp1, dp2, dp3, dp4,
                            angle,
                            target_list,
                            alpha,
                            vstd=None, vave=None, pstd=None, pave=None):
    """ Calculate the best translation for the query map given a rotation angle

    Args:
        data (numpy.array): The data of query map
        vec (numpy.array): The vector representation of query map
        dp1-dp4 (numpy.array): The probability representation of query map
        angle (list/tuple): The rotation angle
        target_list (numpy.array) : A list of FFT-transformed results of the target map
        alpha (float): Parameter for alpha mixing during dot score calculation

    Returns:
        vec_score (float): Best vector dot product score calculated using FFT
        vec_trans (list): Best translation in [x,y,z] with vec_score
        prob_score (float): Best probability score calculated using FFT
        prob_trans (list): Best translation in [x,y,z] with prob_score
    """

    # Rotate the query map and vector representation
    new_vec_array, new_data_array, new_data_array_p1, new_data_array_p2, new_data_array_p3, new_data_array_p4 = rot_mrc_prob(
        data, vec, dp1, dp2, dp3, dp4, angle)

    # Compose the query FFT list

    x2 = new_vec_array[..., 0]
    y2 = new_vec_array[..., 1]
    z2 = new_vec_array[..., 2]

    p21 = new_data_array_p1
    p22 = new_data_array_p2
    p23 = new_data_array_p3
    p24 = new_data_array_p4

    query_list_vec = [x2, y2, z2]
    query_list_prob = [p21, p22, p23, p24]

    # fftw plans initialization
    a = pyfftw.empty_aligned((x2.shape), dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned((x2.shape), dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    # Search for best translation using FFT
    fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec, a, b, c, fft_object, ifft_object)
    fft_result_list_prob = fft_search_best_dot(target_list[3:], query_list_prob, a, b, c, fft_object, ifft_object)

    vec_score, vec_trans = find_best_trans_list(fft_result_list_vec)
    prob_score, prob_trans = find_best_trans_list(fft_result_list_prob)

    mixed_score, mixed_trans = None, None

    if vstd is not None and vave is not None and pstd is not None and pave is not None:
        mixed_score, mixed_trans = find_best_trans_mixed(fft_result_list_vec,
                                                         fft_result_list_prob,
                                                         alpha,
                                                         vstd, vave,
                                                         pstd, pave)

    return vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans


def find_best_trans_mixed(vec_fft_results, prob_fft_results, alpha, vstd, vave, pstd, pave):
    sum_arr_v = sum(vec_fft_results)
    sum_arr_p = sum(prob_fft_results)

    # z-score normalization
    sum_arr_v = (sum_arr_v - vave) / vstd
    sum_arr_p = (sum_arr_p - pave) / pstd

    # alpha mixing
    sum_arr_mixed = (1 - alpha) * sum_arr_v + alpha * sum_arr_p

    # find the best translation
    best_score = sum_arr_mixed.max()
    trans = np.unravel_index(sum_arr_mixed.argmax(), sum_arr_mixed.shape)

    return best_score, trans


# convert euler angles to rotation matrix
def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    return r


def convert_trans(cen1, cen2, r, trans, xwidth2, dim):
    trans = np.array(trans)

    if trans[0] > 0.5 * dim:
        trans[0] -= dim
    if trans[1] > 0.5 * dim:
        trans[1] -= dim
    if trans[2] > 0.5 * dim:
        trans[2] -= dim

    cen2 = r.apply(cen2)  # rotate the center
    new_trans = cen1 - (cen2 + trans * xwidth2)  # calculate new translation
    return new_trans
