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
    non_zero_vec = orig_mrc_vec[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]
    non_zero_dens = orig_mrc_data[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]

    # rotate the vectors
    new_vec = np.einsum("ij, kj->ki", mtx, non_zero_vec)

    # find the new indices
    new_ind_arr = (non_zero_rot_list[:, 3:6] + cent).astype(np.int32)
    # new_ind_arr = (new_pos[dens_mask] + cent).astype(np.int32)

    # fill in the values to new vec and dens array
    new_vec_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = new_vec
    new_data_array[
        new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]
    ] = non_zero_dens

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
    non_zero_vec = vec[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]
    non_zero_dens = data[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]
    non_zero_dens_p1 = prob_c1[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]
    non_zero_dens_p2 = prob_c2[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]
    non_zero_dens_p3 = prob_c3[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]
    non_zero_dens_p4 = prob_c4[
        non_zero_rot_list[:, 0], non_zero_rot_list[:, 1], non_zero_rot_list[:, 2]
    ]

    # find the new indices
    new_ind_arr = (non_zero_rot_list[:, 3:6] + cent).astype(int)

    # save the rotated data
    new_vec_array[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = np.einsum(
        "ij, kj->ki", mtx, non_zero_vec
    )
    new_data_array[
        new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]
    ] = non_zero_dens
    new_p1[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p1
    new_p2[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p2
    new_p3[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p3
    new_p4[new_ind_arr[:, 0], new_ind_arr[:, 1], new_ind_arr[:, 2]] = non_zero_dens_p4

    return new_vec_array, new_data_array, new_p1, new_p2, new_p3, new_p4
