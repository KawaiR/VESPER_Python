import copy

import numpy as np


def rot_mrc(orig_mrc_data, orig_mrc_vec, mtx):
    """A function to rotation the density and vector array by a specified angle.
    Args:
        orig_mrc_data (numpy.array): the data array to be rotated
        orig_mrc_vec (numpy.array): the vector array to be rotated
        mtx (numpy.array): the rotation matrix to be used
    Returns:
        new_vec_array (numpy.array): rotated vector array
        new_data_array (numpy.array): rotated data array
    """

    # set the dimension to be x dimension as all dimension are the same
    dim = orig_mrc_data.shape[0]

    # create array for the positions after rotation
    new_pos = np.array(
        np.meshgrid(
            np.arange(dim),
            np.arange(dim),
            np.arange(dim),
        )
    ).T.reshape(-1, 3)

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


def rot_mrc_prob(data, vec, prob_c1, prob_c2, prob_c3, prob_c4, mtx):
    dim = data.shape[0]

    new_pos = np.array(
        np.meshgrid(
            np.arange(dim),
            np.arange(dim),
            np.arange(dim),
        )
    ).T.reshape(-1, 3)

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
