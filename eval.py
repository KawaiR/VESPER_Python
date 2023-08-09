import copy

import numpy as np
from pyfftw import pyfftw

from search import fft_search_best_dot
from utils import new_rot_mrc_prob, new_rot_mrc

from scipy.spatial.transform import Rotation as R


def eval_score_orig(mrc_target, mrc_search, angle, trans, dot_score_ave, dot_score_std):
    trans = [int(i) for i in trans]

    # Function to evaluate the DOT score for input rotation angle and translation

    # init rotation grid
    search_pos_grid = (
        np.mgrid[
            0 : mrc_search.data.shape[0],
            0 : mrc_search.data.shape[0],
            0 : mrc_search.data.shape[0],
        ]
        .reshape(3, -1)
        .T
    )

    # init the target map vectors
    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
    z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    rd3 = 1.0 / mrc_target.data.size

    # init fft transformation for the target map

    X1 = np.fft.rfftn(x1)
    X1 = np.conj(X1)
    Y1 = np.fft.rfftn(y1)
    Y1 = np.conj(Y1)
    Z1 = np.fft.rfftn(z1)
    Z1 = np.conj(Z1)
    target_list = [X1, Y1, Z1]

    # fftw plans initialization
    a = pyfftw.empty_aligned(mrc_search.vec[..., 0].shape, dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned(mrc_search.vec[..., 0].shape, dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    # init the rotation matrix by euler angle
    rot_mtx = R.from_euler("xyz", angle, degrees=True).as_matrix()

    new_vec, new_data = new_rot_mrc(mrc_search.data, mrc_search.vec, rot_mtx, search_pos_grid)

    x2 = new_vec[..., 0]
    y2 = new_vec[..., 1]
    z2 = new_vec[..., 2]

    query_list_vec = [x2, y2, z2]

    fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec, a, b, c, fft_object, ifft_object)

    sum_arr = np.zeros_like(fft_result_list_vec[0])
    for arr in fft_result_list_vec:
        sum_arr = sum_arr + arr

    dot_score = sum_arr[trans[0]][trans[1]][trans[2]] * rd3
    dot_score = (dot_score - dot_score_ave) / dot_score_std

    return dot_score


def eval_score_mix(
    mrc_target,
    mrc_input,
    mrc_P1,
    mrc_P2,
    mrc_P3,
    mrc_P4,
    mrc_search_p1,
    mrc_search_p2,
    mrc_search_p3,
    mrc_search_p4,
    angle_list,
    trans_list,
    vstd,
    vave,
    pstd,
    pave,
    mix_score_ave,
    mix_score_std,
    alpha,
):
    # convert the translation list to integer
    for trans in trans_list:
        trans = [int(i) for i in trans]

    # init the target map vectors
    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])
    y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
    z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])

    p1 = copy.deepcopy(mrc_P1.data)
    p2 = copy.deepcopy(mrc_P2.data)
    p3 = copy.deepcopy(mrc_P3.data)
    p4 = copy.deepcopy(mrc_P4.data)

    # Score normalization constant

    rd3 = 1.0 / (mrc_target.xdim**3)

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

    # fftw plans initialization
    a = pyfftw.empty_aligned(mrc_search_p1.data.shape, dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned(mrc_search_p1.data.shape, dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    # init rotation grid
    search_pos_grid = (
        np.mgrid[
            0 : mrc_input.data.shape[0],
            0 : mrc_input.data.shape[0],
            0 : mrc_input.data.shape[0],
        ]
        .reshape(3, -1)
        .T
    )

    mix_score_list = []

    for angle, trans in zip(angle_list, trans_list):
        trans = [int(i) for i in trans]

        rot_mtx = R.from_euler("xyz", angle, degrees=True).as_matrix()

        r_vec, r_data, rp1, rp2, rp3, rp4 = new_rot_mrc_prob(
            mrc_input.data,
            mrc_input.vec,
            mrc_search_p1.data,
            mrc_search_p2.data,
            mrc_search_p3.data,
            mrc_search_p4.data,
            rot_mtx,
            search_pos_grid,
        )

        # extract XYZ components from vector representation
        x2 = r_vec[..., 0]
        y2 = r_vec[..., 1]
        z2 = r_vec[..., 2]

        p21 = rp1
        p22 = rp2
        p23 = rp3
        p24 = rp4

        query_list_vec = [x2, y2, z2]
        query_list_prob = [p21, p22, p23, p24]

        fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec, a, b, c, fft_object, ifft_object)
        fft_result_list_prob = fft_search_best_dot(target_list[3:], query_list_prob, a, b, c, fft_object, ifft_object)

        sum_arr_v = fft_result_list_vec[0] + fft_result_list_vec[1] + fft_result_list_vec[2]

        sum_arr_p = fft_result_list_prob[0] + fft_result_list_prob[1] + fft_result_list_prob[2]

        sum_arr_v = (sum_arr_v - vave) / vstd
        sum_arr_p = (sum_arr_p - pave) / pstd

        sum_arr_mixed = (1 - alpha) * sum_arr_v + alpha * sum_arr_p

        best_score = sum_arr_mixed.max()
        best_trans = np.unravel_index(sum_arr_mixed.argmax(), sum_arr_mixed.shape)

        # print(best_score, best_trans)

        mix_score = sum_arr_mixed[trans[0]][trans[1]][trans[2]]

        # print(mix_score)

        mix_score = (mix_score - mix_score_ave) / mix_score_std

        mix_score_list.append(mix_score)

        # print(f"vave={vave}, vstd={vstd}, pave={pave}, pstd={pstd}, mix_score_ave={mix_score_ave}, mix_score_std={mix_score_std}")

    return mix_score_list
