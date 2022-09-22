import copy

import numba
import numpy as np
import pyfftw
from scipy.interpolate import RegularGridInterpolator


def mrc_set_vox_size(mrc, thr=0.00, voxel_size=7.0):
    """Set the voxel size for the specified MrcObj

    Args:
        mrc (MrcObj): [the target MrcObj to set the voxel size for]
        thr (float, optional): preset threshold for density cutoff. Defaults to 0.
        voxel_size (float, optional): the granularity of the voxel in terms of angstroms. Defaults to 7.0.

    Returns:
        mrc (MrcObj): the original MrcObj
        mrc_new (MrcObj): a processed MrcObj
    """

    # if th < 0 add th to all value
    if thr < 0:
        mrc.dens = mrc.dens - thr
        thr = 0.0

    # zero all the values less than threshold
    mrc.dens[mrc.dens < thr] = 0.0
    mrc.data[mrc.data < thr] = 0.0

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
    """Calculate the density and vector in resized map using Gaussian interpolation in original MRC density map"""
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


@numba.jit(nopython=True)
def calc_avg(stp, endp, prob_data, mrc_data):
    """
    It calculates the weighted average probability of a given region of the probability map

    :param stp: start point of the region
    :param endp: the end point of the region of interest
    :param prob_data: the probability data from the .mrc file
    :param mrc_data: the density map
    :return: The average probability of the selected region.
    """
    selected = prob_data[stp[0]:endp[0], stp[1]:endp[1], stp[2]:endp[2]]
    weights = mrc_data[stp[0]:endp[0], stp[1]:endp[1], stp[2]:endp[2]]
    # return 0 if no density in the selected region
    if np.sum(weights) == 0:
        return 0.0
    # return the weighted average probability of the selected region
    return np.average(selected, weights=weights)


def fastVEC(src, dest, dreso=16.0, prob_map=False, density_map=None):
    if prob_map is True and density_map is None:
        print("Error: density_map is not defined")
        exit(-1)

    src_xwidth = src.xwidth
    src_orig = src.orig
    src_dims = np.array((src.xdim, src.ydim, src.zdim))
    dest_xwidth = dest.xwidth
    dest_orig = dest.orig
    dest_dims = np.array((dest.xdim, dest.ydim, dest.zdim))

    # type cast to ensure function signature match
    dest_data, dest_vec = doFastVEC(src_xwidth, src_orig, src_dims.astype(np.int32), src.data,
                                    dest_xwidth, dest_orig, dest_dims.astype(np.int32),
                                    dest.vec, dest.data, dreso, prob_map, density_map)

    dsum = np.sum(dest_data)
    Nact = np.count_nonzero(dest_data)

    # gracefully handle the case where no active voxels are found
    if Nact == 0:
        print("Error: No density value after resampling. Is the voxel spacing parameter too large?")
        exit(-1)

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


# original function signature for linux platform
# @numba.njit((numba.float64, numba.float32[:], numba.int64[:], numba.float32[:, :, :], numba.float64, numba.float64[:],
#              numba.int64[:], numba.float32[:, :, :, :], numba.float32[:, :, :], numba.float64, numba.boolean,
#              numba.float32[:, :, :]), parallel=True)
@numba.njit((numba.float64, numba.float32[:], numba.int32[:], numba.float32[:, :, :], numba.float64, numba.float64[:],
             numba.int32[:], numba.float32[:, :, :, :], numba.float32[:, :, :], numba.float64, numba.boolean,
             numba.float32[:, :, :]))
def doFastVEC(src_xwidth, src_orig, src_dims, src_data,
              dest_xwidth, dest_orig, dest_dims, dest_vec, dest_data,
              dreso=16.0, prob_map=False, density_map=None):
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

                if prob_map:
                    # calculate weighted average over the region
                    dest_data[x][y][z] = calc_avg(stp, endp, src_data, density_map)
                else:
                    # compute the total density and vector with Gaussian weight
                    dtotal, pos2 = calc(stp, endp, pos, src_data, fsiv)
                    dest_data[x][y][z] = dtotal
                    if dtotal == 0:
                        continue

                    # vector normalization
                    rd = 1.0 / dtotal
                    pos2 *= rd
                    tmpcd = pos2 - pos
                    dvec = np.sqrt(tmpcd[0] ** 2 + tmpcd[1] ** 2 + tmpcd[2] ** 2)

                    if dvec == 0:
                        dvec = 1.0

                    rdvec = 1.0 / dvec

                    dest_vec[x][y][z] = tmpcd * rdvec

    return dest_data, dest_vec


# @numba.jit(nopython=True)
# def rot_pos_mtx(mtx, vec):
#     """Rotate a vector or matrix using a rotation matrix.
#
#     Args:
#         mtx (numpy.array): the rotation matrix
#         vec (numpy.array): the vector/matrix to be rotated
#
#     Returns:
#         ret (numpy.array): the rotated vector/matrix
#     """
#     mtx = mtx.astype(np.float32)
#     vec = vec.astype(np.float32)
#
#     ret = vec @ mtx
#
#     return ret


def rot_mrc(orig_mrc_data, orig_mrc_vec, mtx, interp=None):
    """A function to rotation the density and vector array by a specified angle.

    Args:
        orig_mrc_data (numpy.array): the data array to be rotated
        orig_mrc_vec (numpy.array): the vector array to be rotated
        mtx (numpy.array): the rotation matrix

    Returns:
        new_vec_array (numpy.array): rotated vector array
        new_data_array (numpy.array): rotated data array
    """

    Nx, Ny, Nz = orig_mrc_data.shape
    x = np.linspace(0, Nx - 1, Nx)
    y = np.linspace(0, Ny - 1, Ny)
    z = np.linspace(0, Nz - 1, Nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    x_center = x[0] + x[-1] / 2
    y_center = y[0] + y[-1] / 2
    z_center = z[0] + z[-1] / 2

    # center the coord
    coor = np.array([xx - x_center, yy - y_center, zz - z_center])
    # apply rotation
    coor_prime = np.tensordot(mtx, coor, axes=((1), (0)))

    # uncenter the coord
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center

    # trim the values outside boundaries
    x_valid1 = xx_prime >= 0
    x_valid2 = xx_prime <= Nx - 1
    y_valid1 = yy_prime >= 0
    y_valid2 = yy_prime <= Ny - 1
    z_valid1 = zz_prime >= 0
    z_valid2 = zz_prime <= Nz - 1

    # get non-zero indicies in original density
    nonzero_dens = orig_mrc_data > 0

    # get voxels with all valid dimensions
    valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2 * nonzero_dens

    # get nonzero positions
    x_valid_idx, y_valid_idx, z_valid_idx = np.where(valid_voxel > 0)

    # create new arrays to store the final result
    new_data_array = np.zeros_like(orig_mrc_data)
    new_vec_array = np.zeros_like(orig_mrc_vec)

    # gather points to be interpolated
    interp_points = np.array(
        [
            xx_prime[x_valid_idx, y_valid_idx, z_valid_idx],
            yy_prime[x_valid_idx, y_valid_idx, z_valid_idx],
            zz_prime[x_valid_idx, y_valid_idx, z_valid_idx],
        ]
    ).T

    if interp is not None:
        # create grid interpolator
        data_w_coor = RegularGridInterpolator((x, y, z), orig_mrc_data, method=interp)
        vec_w_coor = RegularGridInterpolator((x, y, z), orig_mrc_vec, method=interp)

        # do interpolation
        interp_result = data_w_coor(interp_points)
        vec_result = vec_w_coor(interp_points)

    else:
        # no interpolation
        interp_result = orig_mrc_data[interp_points[:, 0].astype(np.int32),
                                      interp_points[:, 1].astype(np.int32),
                                      interp_points[:, 2].astype(np.int32)]
        vec_result = orig_mrc_vec[interp_points[:, 0].astype(np.int32),
                                  interp_points[:, 1].astype(np.int32),
                                  interp_points[:, 2].astype(np.int32)]

    # save interpolated data
    new_data_array[x_valid_idx, y_valid_idx, z_valid_idx] = interp_result
    new_vec_array[x_valid_idx, y_valid_idx, z_valid_idx] = np.swapaxes(np.tensordot(mtx, np.swapaxes(vec_result, 0, 1),
                                                                                    axes=((0), (0))), 0, 1)

    return new_vec_array, new_data_array


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


def save_pdb(origin,
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


@numba.jit(forceobj=True)
def rot_mrc_prob(data, vec, prob_c1, prob_c2, prob_c3, prob_c4, mtx, interp=None):
    """
    It takes in a 3D array, and a 3x3 rotation matrix, and returns a 3D array that is the result of rotating the input array
    by the rotation matrix

    :param data: the density map
    :param vec: the vector field
    :param prob_c1, prob_c2, prob_c3, prob_c4: probability of 4 classes
    :param mtx: the rotation matrix
    :param interp: interpolation method, defaults to None (optional)
    :return: The new_vec_array is the new vector array after rotation. The new_data_array is the new density array after
    rotation. The new_p1, new_p2, new_p3, new_p4 are the new probability arrays after rotation.
    """

    Nx, Ny, Nz = data.shape
    x = np.linspace(0, Nx - 1, Nx)
    y = np.linspace(0, Ny - 1, Ny)
    z = np.linspace(0, Nz - 1, Nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    x_center = x[0] + x[-1] / 2
    y_center = y[0] + y[-1] / 2
    z_center = z[0] + z[-1] / 2

    # center the coord
    coor = np.array([xx - x_center, yy - y_center, zz - z_center])
    # apply rotation
    coor_prime = np.tensordot(mtx, coor, axes=((1), (0)))

    # uncenter the coord
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center

    # trim the values outside boundaries
    x_valid1 = xx_prime >= 0
    x_valid2 = xx_prime <= Nx - 1
    y_valid1 = yy_prime >= 0
    y_valid2 = yy_prime <= Ny - 1
    z_valid1 = zz_prime >= 0
    z_valid2 = zz_prime <= Nz - 1

    nonzero_dens = data > 0

    # get voxels with all valid dimensions
    valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2 * nonzero_dens

    # get nonzero positions
    x_valid_idx, y_valid_idx, z_valid_idx = np.where(valid_voxel > 0)

    # create new arrays to store the final result
    new_data_array = np.zeros_like(data)
    new_vec_array = np.zeros_like(vec)
    new_p1 = np.zeros_like(prob_c1)
    new_p2 = np.zeros_like(prob_c2)
    new_p3 = np.zeros_like(prob_c3)
    new_p4 = np.zeros_like(prob_c4)

    # gather points to be interpolated
    interp_points = np.array(
        [
            xx_prime[x_valid_idx, y_valid_idx, z_valid_idx],
            yy_prime[x_valid_idx, y_valid_idx, z_valid_idx],
            zz_prime[x_valid_idx, y_valid_idx, z_valid_idx],
        ]
    ).T

    if interp is not None:
        # interpolate
        data_w_coor = RegularGridInterpolator((x, y, z), data, method=interp)
        vec_w_coor = RegularGridInterpolator((x, y, z), vec, method=interp)
        p1_w_coor = RegularGridInterpolator((x, y, z), prob_c1, method=interp)
        p2_w_coor = RegularGridInterpolator((x, y, z), prob_c2, method=interp)
        p3_w_coor = RegularGridInterpolator((x, y, z), prob_c3, method=interp)
        p4_w_coor = RegularGridInterpolator((x, y, z), prob_c4, method=interp)

        interp_result = data_w_coor(interp_points)
        vec_result = vec_w_coor(interp_points)
        p1_result = p1_w_coor(interp_points)
        p2_result = p2_w_coor(interp_points)
        p3_result = p3_w_coor(interp_points)
        p4_result = p4_w_coor(interp_points)

    else:
        # use casting
        interp_result = data[interp_points[:, 0].astype(np.int32),
                             interp_points[:, 1].astype(np.int32),
                             interp_points[:, 2].astype(np.int32)]
        vec_result = vec[interp_points[:, 0].astype(np.int32),
                         interp_points[:, 1].astype(np.int32),
                         interp_points[:, 2].astype(np.int32)]
        p1_result = prob_c1[interp_points[:, 0].astype(np.int32),
                            interp_points[:, 1].astype(np.int32),
                            interp_points[:, 2].astype(np.int32)]
        p2_result = prob_c2[interp_points[:, 0].astype(np.int32),
                            interp_points[:, 1].astype(np.int32),
                            interp_points[:, 2].astype(np.int32)]
        p3_result = prob_c3[interp_points[:, 0].astype(np.int32),
                            interp_points[:, 1].astype(np.int32),
                            interp_points[:, 2].astype(np.int32)]
        p4_result = prob_c4[interp_points[:, 0].astype(np.int32),
                            interp_points[:, 1].astype(np.int32),
                            interp_points[:, 2].astype(np.int32)]

    # save interpolated data
    new_data_array[x_valid_idx, y_valid_idx, z_valid_idx] = interp_result
    new_vec_array[x_valid_idx, y_valid_idx, z_valid_idx] = np.swapaxes(np.tensordot(mtx, np.swapaxes(vec_result, 0, 1),
                                                                                    axes=((0), (0))), 0, 1)
    new_p1[x_valid_idx, y_valid_idx, z_valid_idx] = p1_result
    new_p2[x_valid_idx, y_valid_idx, z_valid_idx] = p2_result
    new_p3[x_valid_idx, y_valid_idx, z_valid_idx] = p3_result
    new_p4[x_valid_idx, y_valid_idx, z_valid_idx] = p4_result

    return new_vec_array, new_data_array, new_p1, new_p2, new_p3, new_p4

    # dim = data.shape[0]
    #
    # new_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), )).T.reshape(-1, 3)
    #
    # cent = 0.5 * float(dim)
    # new_pos = new_pos - cent
    #
    # old_pos = rot_pos_mtx(np.flip(mtx).T, new_pos) + cent
    #
    # combined_arr = np.hstack((old_pos, new_pos))
    #
    # in_bound_mask = (
    #         (old_pos[:, 0] >= 0)
    #         & (old_pos[:, 1] >= 0)
    #         & (old_pos[:, 2] >= 0)
    #         & (old_pos[:, 0] < dim)
    #         & (old_pos[:, 1] < dim)
    #         & (old_pos[:, 2] < dim)
    # )
    #
    # # get the mask of all the values inside boundary
    # new_pos = (new_pos[in_bound_mask] + cent).astype(np.int32)
    #
    # # get the old index array
    # old_pos = old_pos[in_bound_mask].astype(np.int32)
    #
    # old_x = old_pos[:, 0]
    # old_y = old_pos[:, 1]
    # old_z = old_pos[:, 2]
    #
    # new_vec = np.tensordot(np.flip(mtx), vec[old_x, old_y, old_z], axes=((0), (1)))
    #
    # new_x = new_pos[:, 0]
    # new_y = new_pos[:, 1]
    # new_z = new_pos[:, 2]
    #
    # # create new array for density, vector and probability
    # new_vec_array = np.zeros_like(vec)
    # new_data_array = np.zeros_like(data)
    # new_p1 = np.zeros_like(prob_c1)
    # new_p2 = np.zeros_like(prob_c2)
    # new_p3 = np.zeros_like(prob_c3)
    # new_p4 = np.zeros_like(prob_c4)
    #
    # # fill in the values to new vec and dens array
    # new_vec_array[new_x, new_y, new_z] = new_vec.T
    # new_data_array[new_x, new_y, new_z] = data[old_x, old_y, old_z]
    # new_p1[new_x, new_y, new_z] = prob_c1[old_x, old_y, old_z]
    # new_p2[new_x, new_y, new_z] = prob_c2[old_x, old_y, old_z]
    # new_p3[new_x, new_y, new_z] = prob_c3[old_x, old_y, old_z]
    # new_p4[new_x, new_y, new_z] = prob_c4[old_x, old_y, old_z]
    #
    # return new_vec_array, new_data_array, new_p1, new_p2, new_p3, new_p4

    # combined_arr = combined_arr[in_bound_mask]
    #
    # combined_arr = combined_arr.astype(np.int32)
    #
    # index_arr = combined_arr[:, 0:3]
    #
    # dens_mask = data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    # dens_mask_p1 = prob_c1[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    # dens_mask_p2 = prob_c2[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    # dens_mask_p3 = prob_c3[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    # dens_mask_p4 = prob_c4[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    #
    # non_zero_rot_list = combined_arr[dens_mask]
    # non_zero_rot_list_p1 = combined_arr[dens_mask_p1]
    # non_zero_rot_list_p2 = combined_arr[dens_mask_p2]
    # non_zero_rot_list_p3 = combined_arr[dens_mask_p3]
    # non_zero_rot_list_p4 = combined_arr[dens_mask_p4]
    #
    # non_zero_vec = vec[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    #
    # non_zero_dens = data[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    #
    # non_zero_dens_p1 = prob_c1[
    #     non_zero_rot_list_p1[:, 0], non_zero_rot_list_p1[:, 1], non_zero_rot_list_p1[:, 2]]
    #
    # non_zero_dens_p2 = prob_c2[
    #     non_zero_rot_list_p2[:, 0], non_zero_rot_list_p2[:, 1], non_zero_rot_list_p2[:, 2]]
    #
    # non_zero_dens_p3 = prob_c3[
    #     non_zero_rot_list_p3[:, 0], non_zero_rot_list_p3[:, 1], non_zero_rot_list_p3[:, 2]]
    #
    # non_zero_dens_p4 = prob_c4[
    #     non_zero_rot_list_p4[:, 0], non_zero_rot_list_p4[:, 1], non_zero_rot_list_p4[:, 2]]
    #
    # new_vec = rot_pos_mtx(np.flip(mtx), non_zero_vec)
    #
    # new_vec_array = np.zeros_like(vec)
    # new_data_array = np.zeros_like(data)
    # new_data_array_p1 = np.zeros_like(prob_c1)
    # new_data_array_p2 = np.zeros_like(prob_c2)
    # new_data_array_p3 = np.zeros_like(prob_c3)
    # new_data_array_p4 = np.zeros_like(prob_c4)
    #
    # for vec, ind, dens in zip(new_vec, (non_zero_rot_list[:, 3:6] + cent).astype(int), non_zero_dens):
    #     new_vec_array[ind[0]][ind[1]][ind[2]][0] = vec[0]
    #     new_vec_array[ind[0]][ind[1]][ind[2]][1] = vec[1]
    #     new_vec_array[ind[0]][ind[1]][ind[2]][2] = vec[2]
    #     new_data_array[ind[0]][ind[1]][ind[2]] = dens
    #
    # for ind, dens in zip((non_zero_rot_list_p1[:, 3:6] + cent).astype(int), non_zero_dens_p1):
    #     new_data_array_p1[ind[0]][ind[1]][ind[2]] = dens
    #
    # for ind, dens in zip((non_zero_rot_list_p2[:, 3:6] + cent).astype(int), non_zero_dens_p2):
    #     new_data_array_p2[ind[0]][ind[1]][ind[2]] = dens
    #
    # for ind, dens in zip((non_zero_rot_list_p3[:, 3:6] + cent).astype(int), non_zero_dens_p3):
    #     new_data_array_p3[ind[0]][ind[1]][ind[2]] = dens
    #
    # for ind, dens in zip((non_zero_rot_list_p4[:, 3:6] + cent).astype(int), non_zero_dens_p4):
    #     new_data_array_p4[ind[0]][ind[1]][ind[2]] = dens

    # return new_vec_array, new_data_array, new_data_array_p1, new_data_array_p2, new_data_array_p3, new_data_array_p4


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
