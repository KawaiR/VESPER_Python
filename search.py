# coding: utf-8
# import concurrent.futures
import multiprocessing
from datetime import datetime
from pathlib import Path

import mrcfile
import pyfftw
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils import *

pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()


class MrcObj:
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
        self.xwidth = mrc.voxel_size.x.item()
        self.ywidth = mrc.voxel_size.y.item()
        self.zwidth = mrc.voxel_size.z.item()

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
        mrc_target (MrcObj): the input target map
        mrc_search (MrcObj): the input query map
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

        exit(0)

    # init the target map vectors
    x1 = copy.deepcopy(mrc_target.vec[:, :, :, 0])

    # Postprocessing for other modes
    if mode == "Overlap":
        x1 = np.where(mrc_target.data > 0, 1.0, 0.0)
    elif mode == "CC":
        x1 = np.where(mrc_target.data > 0, mrc_target.data, 0.0)
    elif mode == "PCC":
        x1 = np.where(mrc_target.data > 0, mrc_target.data - mrc_target.ave, 0.0)
    elif mode == "Laplacian":
        x1 = laplacian_filter(mrc_target.data)

    rd3 = 1.0 / mrc_target.data.size

    X1 = np.fft.rfftn(x1)
    X1 = np.conj(X1)

    # calculate combination of rotation angles
    angle_comb = calc_angle_comb(ang)

    target_list = [X1]

    # init fft transformation for the target map
    if mode == "VecProduct":
        y1 = copy.deepcopy(mrc_target.vec[:, :, :, 1])
        z1 = copy.deepcopy(mrc_target.vec[:, :, :, 2])
        Y1 = np.fft.rfftn(y1)
        Y1 = np.conj(Y1)
        Z1 = np.fft.rfftn(z1)
        Z1 = np.conj(Z1)
        target_list = [X1, Y1, Z1]

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
        r = R.from_euler('xyz', result_mrc["angle"], degrees=True)
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

        refine_ang_list = np.concatenate(refine_ang_list, axis=0)

        # rotate the mrc vector and data according to the list (multithreaded)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() + 4) as executor:
        #     trans_vec = {executor.submit(rot_mrc, mrc_search.data, mrc_search.vec, angle, ): angle for angle in
        #                  refine_ang_arr}
        #     for future in concurrent.futures.as_completed(trans_vec):
        #         angle = trans_vec[future]
        #         rot_vec_dict[tuple(angle)] = future.result()[0]
        #         rot_data_dict[tuple(angle)] = future.result()[1]

        for angle in tqdm(refine_ang_list, desc="Refining Rotation"):
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
        euler = result_mrc["angle"]
        r = R.from_euler('xyz', euler, degrees=True)
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
        mrc_target (MrcObj): the input target map
        mrc_search (MrcObj): the input query map
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
            vstd=vstd, vave=vave, pstd=pstd, pave=pave)

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

    # sort the list and save topN
    sorted_top_n = sorted(angle_score, key=lambda x: x["mixed_score"], reverse=True)[:TopN]

    # print TopN statistics
    for idx, x in enumerate(sorted_top_n):
        print("M", str(idx + 1), x)

    # 5 degrees local refinement search
    refined_score = []
    if ang > 5.0:
        # Search for +5.0 and -5.0 degree rotation.
        print("\n###Start Refining###")

        refine_ang_list = []
        for result_mrc in sorted_top_n:
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

            # make sure the angles are in the range of 0-360
            ang_list[ang_list < 0] += 360
            # compose the list
            refine_ang_list.append(ang_list)

        refine_ang_list = np.concatenate(refine_ang_list, axis=0)

        for ang in tqdm(refine_ang_list, desc="Local Refining"):
            vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans, r_vec, r_data = rot_and_search_fft_prob(
                mrc_search.data,
                mrc_search.vec,
                mrc_search_p1.data,
                mrc_search_p2.data,
                mrc_search_p3.data,
                mrc_search_p4.data,
                ang,
                target_list,
                alpha,
                ret_data=True,
                vstd=vstd, vave=vave, pstd=pstd, pave=pave)

            refined_score.append(
                {"angle": tuple(ang),
                 "vec_score": vec_score * rd3,
                 "vec_trans": vec_trans,
                 "vec": r_vec,
                 "data": r_data,
                 "prob_score": prob_score * rd3,
                 "prob_trans": prob_trans,
                 "mixed_score": mixed_score * rd3,
                 "mixed_trans": mixed_trans})

        refined_list = sorted(refined_score, key=lambda x: x["mixed_score"], reverse=True)[:TopN]  # Sort by mixed score
    else:
        refined_list = sorted_top_n

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
        r = R.from_euler('xyz', result_mrc["angle"], degrees=True)
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


def rot_and_search_fft(data, vec, angle, target_list, mrc_target, mode="VecProduct"):
    """
    It rotates the query map and vector representation, then searches for the best translation using FFT
    :param data: the 3D map to be rotated
    :param vec: the vector representation of the query map
    :param angle: the angle of rotation
    :param target_list: a list of 3D arrays, each of which is the FFT of a map in the target set
    :param mrc_target: the target map
    :param mode: "VecProduct" or "CC" or "PCC", defaults to VecProduct (optional)
    :return: The score, the translation, the rotated vector, and the rotated data.
    """
    # init the rotation matrix by euler angle
    rot_mtx = R.from_euler("ZYX", angle, degrees=True).as_matrix()

    # Rotate the query map and vector representation
    new_vec, new_data = rot_mrc(data, vec, rot_mtx)

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
                            ret_data=False,
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

    rot_mtx = R.from_euler("ZYX", angle, degrees=True).as_matrix()

    # Rotate the query map and vector representation
    r_vec, r_data, rp1, rp2, rp3, rp4 = \
        rot_mrc_prob(data, vec, dp1, dp2, dp3, dp4, rot_mtx)

    # Compose the query FFT list

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

    if ret_data:
        return vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans, r_vec, r_data
    return vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans


def find_best_trans_mixed(vec_fft_results, prob_fft_results, alpha, vstd, vave, pstd, pave):
    """
    It takes the sum of the two arrays, normalizes them, mixes them, and then finds the best translation
    :param vec_fft_results: the results of the FFT on the vectorized image
    :param prob_fft_results: the FFT of the probability map
    :param alpha: the weight of the probability map
    :param vstd: standard deviation of the vector fft results
    :param vave: the average of the vector fft results
    :param pstd: standard deviation of the probability distribution
    :param pave: the average of the probability array
    :return: The best score and the translation that produced it.
    """
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
