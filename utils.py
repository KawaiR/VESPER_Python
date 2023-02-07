import copy

import numba
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation as R

interp = None


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
    if len(non_zero_index_list) == 0:
        dmax = 0
    else:
        cent_arr = np.array(mrc.cent)
        d2_list = np.linalg.norm(non_zero_index_list - cent_arr, axis=1)
        dmax = np.max(d2_list)

    print()
    print("#dmax=" + str(dmax / mrc.xwidth))

    # set new center
    new_cent = mrc.cent * mrc.xwidth + mrc.orig

    tmp_size = 2 * dmax * mrc.xwidth / voxel_size

    # get the best size suitable for fft operation
    # new_xdim = pyfftw.next_fast_len(int(tmp_size))

    a = 2
    while (1):
        if a > tmp_size:
            break
        a *= 2

    b = 3
    while (1):
        if b > tmp_size:
            break
        b *= 2

    b = 9
    while (1):
        if b > tmp_size:
            break
        b *= 2
    if a > b:
        new_xdim = b
    else:
        new_xdim = a

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


# @numba.jit(nopython=True)
# def calc(stp, endp, pos, mrc1_data, fsiv):
#     """Vectorized version of calc"""
#
#     xx = np.arange(stp[0], endp[0], 1)
#     yy = np.arange(stp[1], endp[1], 1)
#     zz = np.arange(stp[2], endp[2], 1)
#
#     xx = np.expand_dims(xx, axis=1)
#     xx = np.expand_dims(xx, axis=1)
#     yy = np.expand_dims(yy, axis=1)
#     yy = np.expand_dims(yy, axis=0)
#     zz = np.expand_dims(zz, axis=0)
#     zz = np.expand_dims(zz, axis=0)
#
#     # calculate the distance between the center of the voxel and the center of the particle
#     d2 = (xx - pos[0])**2 + (yy - pos[1])**2 + (zz - pos[2])**2
#
#     # calculate the density and vector in resized map using Gaussian interpolation in original MRC density map
#     d = np.exp(- 1.5 * d2 * fsiv) * mrc1_data[stp[0]:endp[0], stp[1]:endp[1], stp[2]:endp[2]]
#     dtotal = np.sum(d)
#
#     # calculate the vector
#     v = np.array([np.sum(d * xx), np.sum(d * yy), np.sum(d * zz)])
#
#     return dtotal, v

@numba.jit(nopython=True)
def calc_prob(stp, endp, pos, density_data, prob_data, fsiv):
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
                # v = density_data[xp][yp][zp] * prob_data[xp][yp][zp] *  np.exp(-1.5 * d2 * fsiv)
                v = prob_data[xp][yp][zp] * np.exp(-1.5 * d2 * fsiv)
                dtotal += v
                pos2[0] += v * xp
                pos2[1] += v * yp
                pos2[2] += v * zp

    return dtotal, pos2


@numba.jit(nopython=True)
def calc(stp, endp, pos, data, fsiv):
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
                v = data[xp][yp][zp] * np.exp(-1.5 * d2 * fsiv)
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


def fastVEC(src, dest, dreso=16.0, density_map=None):
    src_dims = np.array((src.xdim, src.ydim, src.zdim))
    dest_dims = np.array((dest.xdim, dest.ydim, dest.zdim))

    if np.sum(src.data) != 0:

        dest_data, dest_vec = doFastVEC(src.xwidth, src.orig, src_dims, src.data,
                                        dest.xwidth, dest.orig, dest_dims, dest.data, dest.vec,
                                        dreso, density_map=density_map)

        # calculate map statistics
        dsum = np.sum(dest_data)
        Nact = np.count_nonzero(dest_data)
        ave = np.mean(dest_data[dest_data > 0])
        std = np.linalg.norm(dest_data[dest_data > 0])
        std_norm_ave = np.linalg.norm(dest_data[dest_data > 0] - ave)



    else:
        dsum = 0
        Nact = 0
        ave = 0
        std = 0
        std_norm_ave = 0
        dest_data = np.zeros(dest_dims)
        dest_vec = np.zeros((dest_dims[0], dest_dims[1], dest_dims[2], 3))

    print("#MAP SUM={sum} COUNT={cnt} AVE={ave} STD={std} STD_norm={std_norm}".format(sum=dsum,
                                                                                      cnt=Nact,
                                                                                      ave=ave,
                                                                                      std=std,
                                                                                      std_norm=std_norm_ave))

    # update the dest object with the new data and vectors
    dest.data = dest_data
    dest.vec = dest_vec
    dest.dsum = dsum
    dest.Nact = Nact
    dest.ave = ave
    dest.std = std
    dest.std_norm_ave = std_norm_ave

    return dest


@numba.jit(nopython=True)
def doFastVEC(src_xwidth, src_orig, src_dims, src_data,
              dest_xwidth, dest_orig, dest_dims, dest_data, dest_vec,
              dreso, density_map):
    gstep = src_xwidth
    fs = (dreso / gstep) * 0.5
    fs = fs ** 2
    fsiv = 1.0 / fs
    fmaxd = (dreso / gstep) * 2.0
    print("#maxd=", fmaxd)
    print("#fsiv=", fsiv)

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
                if density_map is not None:
                    dtotal, pos2 = calc_prob(stp, endp, pos, density_map, src_data, fsiv)
                else:
                    dtotal, pos2 = calc(stp, endp, pos, src_data, fsiv)

                if dtotal == 0:
                    continue

                dest_data[x][y][z] = dtotal

                rd = 1.0 / dtotal

                pos2 *= rd

                tmpcd = pos2 - pos

                dvec = np.sqrt(tmpcd[0] ** 2 + tmpcd[1] ** 2 + tmpcd[2] ** 2)

                if dvec == 0:
                    dvec = 1.0

                rdvec = 1.0 / dvec

                dest_vec[x][y][z] = tmpcd * rdvec

    return dest_data, dest_vec


def new_rot_mrc(orig_mrc_data, orig_mrc_vec, mtx, new_pos_grid):

    # set the dimension to be x dimension as all dimension are the same
    dim = orig_mrc_data.shape[0]

    # set the rotation center
    cent = 0.5 * float(dim)

    # get relative new positions from center
    new_pos = new_pos_grid - cent

    # reversely rotate the new position lists to get old positions
    old_pos = np.einsum("ij, kj->ki", mtx.T, new_pos) + cent
    #old_pos = new_pos @ mtx + 0.5 * float(dim)

    # init new vec and dens array
    new_vec_array = np.zeros_like(orig_mrc_vec)
    new_data_array = np.zeros_like(orig_mrc_data)

    in_bound_mask = (
            (old_pos[:, 0] >= 0)
            * (old_pos[:, 1] >= 0)
            * (old_pos[:, 2] >= 0)
            * (old_pos[:, 0] < dim)
            * (old_pos[:, 1] < dim)
            * (old_pos[:, 2] < dim)
    )

    # get valid old positions in bound
    valid_old_pos = (old_pos[in_bound_mask]).astype(np.int32)

    # get nonzero density positions in the map
    non_zero_mask = orig_mrc_data[valid_old_pos[:, 0], valid_old_pos[:, 1], valid_old_pos[:, 2]] > 0

    # apply nonzero mask to valid positions
    non_zero_old_pos = valid_old_pos[non_zero_mask]

    # get corresponding new positions
    new_pos = (new_pos[in_bound_mask][non_zero_mask] + cent).astype(np.int32)

    # fill new density entries
    new_data_array[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = orig_mrc_data[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]

    # fetch and rotate the vectors
    non_zero_vecs = orig_mrc_vec[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]

    #new_vec = non_zero_vecs @ mtx.T
    new_vec = np.einsum("ij, kj->ki", mtx, non_zero_vecs)

    # fill new vector entries
    new_vec_array[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = new_vec

    return new_vec_array, new_data_array


def rot_mrc(orig_mrc_data, orig_mrc_vec, mtx):
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
    dim = orig_mrc_data.shape[0]

    # create array for the positions after rotation
    new_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), )).T.reshape(-1, 3)

    # set the rotation center
    cent = 0.5 * float(dim)

    # get relative new positions from center
    new_pos = new_pos - cent

    # reversely rotate the new position lists to get old positions
    old_pos = np.einsum("ij, kj->ki", mtx.T, new_pos) + cent

    # concatenate combine two position array horizontally for later filtering
    combined_arr = np.hstack((old_pos, new_pos))

    # filter out the positions that are out of the original array
    in_bound_mask = (
            (combined_arr[:, 0] >= 0)
            * (combined_arr[:, 1] >= 0)
            * (combined_arr[:, 2] >= 0)
            * (combined_arr[:, 0] < dim)
            * (combined_arr[:, 1] < dim)
            * (combined_arr[:, 2] < dim)
    )

    # init new vec and dens array
    new_vec_array = np.zeros_like(orig_mrc_vec)
    new_data_array = np.zeros_like(orig_mrc_data)

    # get the mask of all the values inside boundary
    combined_arr = combined_arr[in_bound_mask]

    # convert the index to integer
    combined_arr = combined_arr.astype(np.int32)

    # get the old index array
    index_arr = combined_arr[:, 0:3]

    # index_arr = old_pos[in_bound_mask].astype(np.int32)

    # get the index that has non-zero density by masking

    dens_mask = orig_mrc_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    non_zero_rot_list = combined_arr[dens_mask]

    # dens_mask = orig_mrc_data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0
    # non_zero_rot_list = old_pos[dens_mask].astype(np.int32)

    # get the non-zero vec and dens values
    non_zero_vec = orig_mrc_vec[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    non_zero_dens = orig_mrc_data[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]

    # rotate the vectors
    new_vec = np.einsum("ij, kj->ki", mtx, non_zero_vec)

    # find the new indices
    new_ind_arr = (non_zero_rot_list[:, 3:6] + cent).astype(np.int32)
    # new_ind_arr = (new_pos[dens_mask] + cent).astype(np.int32)

    # fill in the values to new vec and dens array
    new_vec_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = new_vec
    new_data_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens

    return new_vec_array, new_data_array


# def rot_mrc(orig_mrc_data, orig_mrc_vec, mtx, interp=interp):
#     """A function to rotation the density and vector array by a specified angle.

#     Args:
#         orig_mrc_data (numpy.array): the data array to be rotated
#         orig_mrc_vec (numpy.array): the vector array to be rotated
#         mtx (numpy.array): the rotation matrix
#         interp (str): the interpolation method

#     Returns:
#         new_vec_array (numpy.array): rotated vector array
#         new_data_array (numpy.array): rotated data array
#     """

#     Nx, Ny, Nz = orig_mrc_data.shape
#     x = np.linspace(0, Nx - 1, Nx)
#     y = np.linspace(0, Ny - 1, Ny)
#     z = np.linspace(0, Nz - 1, Nz)
#     # xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
#     zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

#     x_center = x[0] + x[-1] / 2
#     y_center = y[0] + y[-1] / 2
#     z_center = z[0] + z[-1] / 2

#     # center the coord
#     coor = np.array([xx - x_center, yy - y_center, zz - z_center])
#     # apply rotation
#     # coor_prime = np.tensordot(np.flip(mtx.T), coor, axes=((0), (1)))
#     coor_prime = np.einsum("il, ljkm->ijkm", mtx.T, coor)

#     # uncenter the coord
#     xx_prime = coor_prime[0] + x_center
#     yy_prime = coor_prime[1] + y_center
#     zz_prime = coor_prime[2] + z_center

#     # trim the values outside boundaries
#     x_valid1 = xx_prime >= 0
#     x_valid2 = xx_prime <= Nx - 1
#     y_valid1 = yy_prime >= 0
#     y_valid2 = yy_prime <= Ny - 1
#     z_valid1 = zz_prime >= 0
#     z_valid2 = zz_prime <= Nz - 1

#     # get non-zero indicies in original density
#     nonzero_dens = orig_mrc_data > 0

#     # get voxels with all valid dimensions
#     valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2 * nonzero_dens

#     # get nonzero positions
#     #x_valid_idx, y_valid_idx, z_valid_idx = np.where(valid_voxel > 0)
#     z_valid_idx, y_valid_idx, x_valid_idx = np.where(valid_voxel > 0)

#     # create new arrays to store the final result
#     new_data_array = np.zeros_like(orig_mrc_data)
#     new_vec_array = np.zeros_like(orig_mrc_vec)

#     # gather points to be interpolated
#     # interp_points = np.array(
#     #     [
#     #         xx_prime[x_valid_idx, y_valid_idx, z_valid_idx],
#     #         yy_prime[x_valid_idx, y_valid_idx, z_valid_idx],
#     #         zz_prime[x_valid_idx, y_valid_idx, z_valid_idx],
#     #     ]
#     # ).T

#     interp_points = np.array(
#         [
#             zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
#             yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
#             xx_prime[z_valid_idx, y_valid_idx, x_valid_idx],
#         ]
#     ).T

#     if interp is not None:
#         # create grid interpolator
#         # data_w_coor = RegularGridInterpolator((x, y, z), orig_mrc_data, method=interp)
#         # vec_w_coor = RegularGridInterpolator((x, y, z), orig_mrc_vec, method=interp)

#         data_w_coor = RegularGridInterpolator((z, y, x), orig_mrc_data, method=interp)
#         vec_w_coor = RegularGridInterpolator((z, y, x), orig_mrc_vec, method=interp)

#         # do interpolation
#         interp_result = data_w_coor(interp_points)
#         vec_result = vec_w_coor(interp_points)

#     else:
#         # no interpolation
#         # interp_result = orig_mrc_data[interp_points[:, 0].astype(np.int32),
#         #                               interp_points[:, 1].astype(np.int32),
#         #                               interp_points[:, 2].astype(np.int32)]
#         # vec_result = orig_mrc_vec[interp_points[:, 0].astype(np.int32),
#         #                           interp_points[:, 1].astype(np.int32),
#         #                           interp_points[:, 2].astype(np.int32)]

#         interp_result = orig_mrc_data[interp_points[:, 2].astype(np.int32),
#                                       interp_points[:, 1].astype(np.int32),
#                                       interp_points[:, 0].astype(np.int32)]
#         vec_result = orig_mrc_vec[interp_points[:, 2].astype(np.int32),
#                                   interp_points[:, 1].astype(np.int32),
#                                   interp_points[:, 0].astype(np.int32)]

#     # save interpolated data

#     new_data_array[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result
#     new_vec_array[z_valid_idx, y_valid_idx, x_valid_idx] = np.einsum("ij, kj->ki", mtx, vec_result)

#     # new_data_array[x_valid_idx, y_valid_idx, z_valid_idx] = interp_result
#     # new_vec_array[x_valid_idx, y_valid_idx, z_valid_idx] = np.einsum("ij, kj->ki", mtx, vec_result)

#     return new_vec_array, new_data_array


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


# @numba.jit(forceobj=True)
# def rot_mrc_prob(data, vec, prob_c1, prob_c2, prob_c3, prob_c4, mtx, interp=interp):
#     """
#     It takes in a 3D array, and a 3x3 rotation matrix, and returns a 3D array that is the result of rotating the input array
#     by the rotation matrix
#
#     :param data: the density map
#     :param vec: the vector field
#     :param prob_c1, prob_c2, prob_c3, prob_c4: probability of 4 classes
#     :param mtx: the rotation matrix
#     :param interp: interpolation method, defaults to None (optional)
#     :return: The new_vec_array is the new vector array after rotation. The new_data_array is the new density array after
#     rotation. The new_p1, new_p2, new_p3, new_p4 are the new probability arrays after rotation.
#     """
#
#     Nx, Ny, Nz = data.shape
#     x = np.linspace(0, Nx - 1, Nx)
#     y = np.linspace(0, Ny - 1, Ny)
#     z = np.linspace(0, Nz - 1, Nz)
#     xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
#
#     x_center = x[0] + x[-1] / 2
#     y_center = y[0] + y[-1] / 2
#     z_center = z[0] + z[-1] / 2
#
#     # center the coord
#     coor = np.array([xx - x_center, yy - y_center, zz - z_center])
#     # apply rotation
#     coor_prime = np.tensordot(mtx, coor, axes=((1), (0)))
#
#     # uncenter the coord
#     xx_prime = coor_prime[0] + x_center
#     yy_prime = coor_prime[1] + y_center
#     zz_prime = coor_prime[2] + z_center
#
#     # trim the values outside boundaries
#     x_valid1 = xx_prime >= 0
#     x_valid2 = xx_prime <= Nx - 1
#     y_valid1 = yy_prime >= 0
#     y_valid2 = yy_prime <= Ny - 1
#     z_valid1 = zz_prime >= 0
#     z_valid2 = zz_prime <= Nz - 1
#
#     nonzero_dens = data > 0
#
#     # get voxels with all valid dimensions
#     valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2 * nonzero_dens
#
#     # get nonzero positions
#     x_valid_idx, y_valid_idx, z_valid_idx = np.where(valid_voxel > 0)
#
#     # create new arrays to store the final result
#     new_data_array = np.zeros_like(data)
#     new_vec_array = np.zeros_like(vec)
#     new_p1 = np.zeros_like(prob_c1)
#     new_p2 = np.zeros_like(prob_c2)
#     new_p3 = np.zeros_like(prob_c3)
#     new_p4 = np.zeros_like(prob_c4)
#
#     # gather points to be interpolated
#     interp_points = np.array(
#         [
#             xx_prime[x_valid_idx, y_valid_idx, z_valid_idx],
#             yy_prime[x_valid_idx, y_valid_idx, z_valid_idx],
#             zz_prime[x_valid_idx, y_valid_idx, z_valid_idx],
#         ]
#     ).T
#
#     if interp is not None:
#         # interpolate
#         data_w_coor = RegularGridInterpolator((x, y, z), data, method=interp)
#         vec_w_coor = RegularGridInterpolator((x, y, z), vec, method=interp)
#         p1_w_coor = RegularGridInterpolator((x, y, z), prob_c1, method=interp)
#         p2_w_coor = RegularGridInterpolator((x, y, z), prob_c2, method=interp)
#         p3_w_coor = RegularGridInterpolator((x, y, z), prob_c3, method=interp)
#         p4_w_coor = RegularGridInterpolator((x, y, z), prob_c4, method=interp)
#
#         interp_result = data_w_coor(interp_points)
#         vec_result = vec_w_coor(interp_points)
#         p1_result = p1_w_coor(interp_points)
#         p2_result = p2_w_coor(interp_points)
#         p3_result = p3_w_coor(interp_points)
#         p4_result = p4_w_coor(interp_points)
#
#     else:
#         # use casting
#         interp_result = data[interp_points[:, 0].astype(np.int32),
#                              interp_points[:, 1].astype(np.int32),
#                              interp_points[:, 2].astype(np.int32)]
#         vec_result = vec[interp_points[:, 0].astype(np.int32),
#                          interp_points[:, 1].astype(np.int32),
#                          interp_points[:, 2].astype(np.int32)]
#         p1_result = prob_c1[interp_points[:, 0].astype(np.int32),
#                             interp_points[:, 1].astype(np.int32),
#                             interp_points[:, 2].astype(np.int32)]
#         p2_result = prob_c2[interp_points[:, 0].astype(np.int32),
#                             interp_points[:, 1].astype(np.int32),
#                             interp_points[:, 2].astype(np.int32)]
#         p3_result = prob_c3[interp_points[:, 0].astype(np.int32),
#                             interp_points[:, 1].astype(np.int32),
#                             interp_points[:, 2].astype(np.int32)]
#         p4_result = prob_c4[interp_points[:, 0].astype(np.int32),
#                             interp_points[:, 1].astype(np.int32),
#                             interp_points[:, 2].astype(np.int32)]
#
#     # save interpolated data
#     new_data_array[x_valid_idx, y_valid_idx, z_valid_idx] = interp_result
#     new_vec_array[x_valid_idx, y_valid_idx, z_valid_idx] = np.swapaxes(np.tensordot(mtx, np.swapaxes(vec_result, 0, 1),
#                                                                                     axes=((0), (0))), 0, 1)
#     new_p1[x_valid_idx, y_valid_idx, z_valid_idx] = p1_result
#     new_p2[x_valid_idx, y_valid_idx, z_valid_idx] = p2_result
#     new_p3[x_valid_idx, y_valid_idx, z_valid_idx] = p3_result
#     new_p4[x_valid_idx, y_valid_idx, z_valid_idx] = p4_result
#
#     return new_vec_array, new_data_array, new_p1, new_p2, new_p3, new_p4


def new_rot_mrc_prob(data, vec, prob_c1, prob_c2, prob_c3, prob_c4, mtx, new_pos_grid):

    dim = data.shape[0]
    cent = 0.5 * float(dim)
    new_pos = new_pos_grid - cent

    # reversely rotate the new position lists to get old positions
    old_pos = np.einsum("ij, kj->ki", mtx.T, new_pos) + cent

    # create new array for density, vector and probability
    new_vec_array = np.zeros_like(vec)
    new_data_array = np.zeros_like(data)
    new_p1 = np.zeros_like(prob_c1)
    new_p2 = np.zeros_like(prob_c2)
    new_p3 = np.zeros_like(prob_c3)
    new_p4 = np.zeros_like(prob_c4)

    in_bound_mask = (
            (old_pos[:, 0] >= 0)
            * (old_pos[:, 1] >= 0)
            * (old_pos[:, 2] >= 0)
            * (old_pos[:, 0] < dim)
            * (old_pos[:, 1] < dim)
            * (old_pos[:, 2] < dim)
    )

    # get valid old positions in bound
    valid_old_pos = (old_pos[in_bound_mask]).astype(np.int32)

    # get nonzero density positions in the map
    non_zero_mask = data[valid_old_pos[:, 0], valid_old_pos[:, 1], valid_old_pos[:, 2]] > 0

    # apply nonzero mask to valid positions
    non_zero_old_pos = valid_old_pos[non_zero_mask]

    # get corresponding new positions
    new_pos = (new_pos[in_bound_mask][non_zero_mask] + cent).astype(np.int32)

    # fill new density entries
    new_data_array[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = data[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]
    new_p1[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = prob_c1[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]
    new_p2[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = prob_c2[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]
    new_p3[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = prob_c3[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]
    new_p4[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = prob_c4[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]

    # fetch and rotate the vectors
    non_zero_vecs = vec[non_zero_old_pos[:, 0], non_zero_old_pos[:, 1], non_zero_old_pos[:, 2]]

    #new_vec = non_zero_vecs @ mtx.T
    new_vec = np.einsum("ij, kj->ki", mtx, non_zero_vecs)

    # fill new vector entries
    new_vec_array[new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]] = new_vec

    return new_vec_array, new_data_array, new_p1, new_p2, new_p3, new_p4


def rot_mrc_prob(data, vec, prob_c1, prob_c2, prob_c3, prob_c4, mtx):
    dim = data.shape[0]

    new_pos = np.array(np.meshgrid(np.arange(dim), np.arange(dim), np.arange(dim), )).T.reshape(-1, 3)

    cent = 0.5 * float(dim)
    new_pos = new_pos - cent

    old_pos = np.einsum("ij, kj->ki", mtx.T, new_pos) + cent

    combined_arr = np.hstack((old_pos, new_pos))

    in_bound_mask = (
            (old_pos[:, 0] >= 0)
            * (old_pos[:, 1] >= 0)
            * (old_pos[:, 2] >= 0)
            * (old_pos[:, 0] < dim)
            * (old_pos[:, 1] < dim)
            * (old_pos[:, 2] < dim)
    )

    # create new array for density, vector and probability
    new_vec_array = np.zeros_like(vec)
    new_data_array = np.zeros_like(data)
    new_p1 = np.zeros_like(prob_c1)
    new_p2 = np.zeros_like(prob_c2)
    new_p3 = np.zeros_like(prob_c3)
    new_p4 = np.zeros_like(prob_c4)

    combined_arr = combined_arr[in_bound_mask]

    combined_arr = combined_arr.astype(np.int32)

    index_arr = combined_arr[:, 0:3]

    dens_mask = data[index_arr[:, 0], index_arr[:, 1], index_arr[:, 2]] != 0.0

    non_zero_rot_list = combined_arr[dens_mask]

    # get the index of the non-zero density
    non_zero_vec = vec[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    non_zero_dens = data[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    non_zero_dens_p1 = prob_c1[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    non_zero_dens_p2 = prob_c2[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    non_zero_dens_p3 = prob_c3[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]
    non_zero_dens_p4 = prob_c4[non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]]

    # find the new indices
    new_ind_arr = (non_zero_rot_list[:, 3:6] + cent).astype(int)

    # save the rotated data
    new_vec_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = np.einsum("ij, kj->ki", mtx, non_zero_vec)
    new_data_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens
    new_p1[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p1
    new_p2[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p2
    new_p3[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p3
    new_p4[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p4

    return new_vec_array, new_data_array, new_p1, new_p2, new_p3, new_p4


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

    print("DOT Score=", sco_sum)
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
    # return angle_comb
    seen = set()
    uniq = []
    for ang in angle_comb:
        quat = tuple(np.round(R.from_euler('ZYX', ang, degrees=True).as_quat(), 4))
        if quat not in seen:
            uniq.append(ang)
            seen.add(quat)

    return uniq


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
