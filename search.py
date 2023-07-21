# coding: utf-8
# import concurrent.futures
import multiprocessing
import os
from datetime import datetime
from pathlib import Path

import mrcfile
import pyfftw
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.ndimage import center_of_mass

from utils import *
from utils import gpu_rot_and_search_fft, find_best_trans_list, format_score_result, find_best_trans_mixed

pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.config.NUM_THREADS = max(multiprocessing.cpu_count() - 2, 2)  # Maybe the CPU is sweating too much?


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


def fft_get_score_trans_other(target_X, search_data, a, b, fft_object, ifft_object, mode, ave=None):
    """1D version of fft_search_score_trans to work with other modes.

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
                   showPDB=False, folder=None, gpu=False, gpu_id=-1, remove_dup=False, ldp_path=None,
                   backbone_path=None, input_pdb=None):
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

    if gpu:
        # set up torch cuda device
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. Please check your CUDA installation.")
            exit(1)
        else:
            # set up torch cuda device
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id} for CUDA acceleration.")
    else:
        print("Using FFTW3 for CPU.")

    # init rotation grid
    search_pos_grid = np.mgrid[0:mrc_search.data.shape[0], 0:mrc_search.data.shape[0],
                      0:mrc_search.data.shape[0]].reshape(3, -1).T

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

    # init fft transformation for the target map
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

    if gpu:
        # convert to tensor on GPU
        import torch  # lazy import
        target_list = [torch.from_numpy(target_list[i]).cuda() for i in range(len(target_list))]
    else:
        # fftw plans initialization
        a = pyfftw.empty_aligned(mrc_search.vec[..., 0].shape, dtype="float32")
        b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
        c = pyfftw.empty_aligned(mrc_search.vec[..., 0].shape, dtype="float32")

        fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
        ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    print("###Start Searching###")

    angle_score = []

    # search process
    for angle in tqdm(angle_comb):
        if gpu:
            vec_score, vec_trans, _, _ = gpu_rot_and_search_fft(mrc_search.data,
                                                                mrc_search.vec,
                                                                angle,
                                                                target_list,
                                                                mrc_target,
                                                                device,
                                                                mode=mode,
                                                                new_pos_grid=search_pos_grid)
        else:
            vec_score, vec_trans, _, _ = rot_and_search_fft(mrc_search.data,
                                                            mrc_search.vec,
                                                            angle,
                                                            target_list,
                                                            mrc_target,
                                                            (a, b, c),
                                                            fft_object,
                                                            ifft_object,
                                                            mode=mode,
                                                            new_pos_grid=search_pos_grid)
        angle_score.append({
            "angle": angle,
            "vec_score": vec_score * rd3,
            "vec_trans": vec_trans,
            "ldp_recall": 0.0,
        })

    # calculate the ave and std
    score_arr = np.array([row["vec_score"] for row in angle_score])
    ave = np.mean(score_arr)
    std = np.std(score_arr)
    print("\nStd= " + str(std) + " Ave= " + str(ave) + "\n")

    if remove_dup:

        print("###Start Duplicate Removal###")

        # duplicate removal
        hash_angs = {}

        non_dup_count = 0

        # at least 30 degrees apart
        n_angles_apart = 30 // ang
        rng = n_angles_apart * int(ang)
        rng = int(rng)

        for i, result_mrc in enumerate(angle_score):
            # duplicate removal
            if tuple(result_mrc["angle"]) in hash_angs.keys():
                # print(f"Duplicate: {result_mrc['angle']}")
                trans = hash_angs[tuple(result_mrc["angle"])]
                # manhattan distance
                if np.sum(np.abs(trans - result_mrc["vec_trans"])) < 30:
                    result_mrc["vec_score"] = 0
                    continue

            # add to hash
            hash_angs[tuple(result_mrc["angle"])] = np.array(result_mrc["vec_trans"])

            angle_x = int(result_mrc["angle"][0])
            angle_y = int(result_mrc["angle"][1])
            angle_z = int(result_mrc["angle"][2])

            # add surrounding angles to hash
            for xx in range(angle_x - rng, angle_x + rng + 1, int(ang)):
                for yy in range(angle_y - rng, angle_y + rng + 1, int(ang)):
                    for zz in range(angle_z - rng, angle_z + rng + 1, int(ang)):
                        x_positive = xx % 360
                        y_positive = yy % 360
                        z_positive = zz % 180

                        x_positive = x_positive + 360 if x_positive < 0 else x_positive
                        y_positive = y_positive + 360 if y_positive < 0 else y_positive
                        z_positive = z_positive + 180 if z_positive < 0 else z_positive

                        curr_trans = np.array([x_positive, y_positive, z_positive]).astype(np.float64)
                        # insert into hash
                        hash_angs[tuple(curr_trans)] = np.array(result_mrc["vec_trans"])

            non_dup_count += 1

        print("#Non-duplicate count: " + str(non_dup_count))

    # LDP Recall calculation and sort

    if ldp_path is not None and backbone_path is not None:

        # get atom coords from ldp
        ldp_atoms = []
        with open(ldp_path) as f:
            for line in f:
                tokens = line.split()
                if tokens[0] == "ATOM":
                    ldp_atoms.append(np.array((float(tokens[6]), float(tokens[7]), float(tokens[8]))))
        ldp_atoms = np.array(ldp_atoms)
        ldp_atoms = torch.from_numpy(ldp_atoms).to(device)
        ldp_atoms = ldp_atoms.unsqueeze(0)

        # get ca atoms from backbone
        backbone_ca = []
        with open(backbone_path) as f:
            for line in f:
                tokens = line.split()
                if tokens[0] == "ATOM" and tokens[2] == "CA":  # only CA atoms
                    # if tokens[0] == "ATOM": # all atoms
                    backbone_ca.append(np.array((float(tokens[6]), float(tokens[7]), float(tokens[8]))))

        backbone_ca = np.array(backbone_ca)
        backbone_ca = torch.from_numpy(backbone_ca).to(device)

        for i, result_mrc in enumerate(angle_score):
            rot_vec = result_mrc["angle"]
            trans_vec = np.array(result_mrc["vec_trans"])
            trans_vec = torch.from_numpy(trans_vec).to(device)

            rot_mtx = R.from_euler('xyz', rot_vec, degrees=True).inv().as_matrix()
            rot_mtx = torch.from_numpy(rot_mtx).to(device)

            # rotated backbone CA
            rot_backbone_ca = torch.matmul(backbone_ca, rot_mtx) + trans_vec
            rot_backbone_ca = rot_backbone_ca.unsqueeze(1)

            # calculate all pairwise distances
            dist_mtx = torch.nn.functional.pairwise_distance(rot_backbone_ca, ldp_atoms, p=2.0)

            # get the min distance for each ldp atom
            min_dist = torch.min(dist_mtx, dim=1)[0]

            # get the number of ca atoms within 3.0 angstroms
            num_ca = torch.sum(min_dist < 3.0)
            num_ca = num_ca.cpu().numpy()

            # calculate the ldp recall
            result_mrc["ldp_recall"] = num_ca / len(rot_backbone_ca)

    # sort the list and get top N results
    if ldp_path is not None and backbone_path is not None:
        sorted_top_n = sorted(angle_score, key=lambda x: x["ldp_recall"], reverse=True)[:TopN]
    else:
        sorted_top_n = sorted(angle_score, key=lambda x: x["vec_score"], reverse=True)[:TopN]

    for i, result_mrc in enumerate(sorted_top_n):
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

    if ang > 5.0:
        print("###Start Refining###")
        refined_list = []
        for result_mrc in sorted_top_n:
            refined_score = []
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

            for angle in tqdm(ang_list, desc="Refining Rotation"):
                if gpu:
                    vec_score, vec_trans, new_vec, new_data = gpu_rot_and_search_fft(mrc_search.data,
                                                                                     mrc_search.vec,
                                                                                     angle,
                                                                                     target_list,
                                                                                     mrc_target,
                                                                                     device,
                                                                                     mode=mode,
                                                                                     new_pos_grid=search_pos_grid)
                else:
                    vec_score, vec_trans, new_vec, new_data = rot_and_search_fft(mrc_search.data,
                                                                                 mrc_search.vec,
                                                                                 angle,
                                                                                 target_list,
                                                                                 mrc_target,
                                                                                 (a, b, c),
                                                                                 fft_object,
                                                                                 ifft_object,
                                                                                 mode=mode,
                                                                                 new_pos_grid=search_pos_grid)
                refined_score.append({"angle": tuple(angle),
                                      "vec_score": vec_score * rd3,
                                      "vec_trans": vec_trans,
                                      "vec": new_vec,
                                      "data": new_data})

            refined_list.append(max(refined_score, key=lambda x: x["vec_score"]))
        refined_list = sorted(refined_list, key=lambda x: x["vec_score"], reverse=True)
    else:
        refined_list = sorted_top_n

    # refined_score = []
    # if ang > 5.0:
    #
    #     # setup all the angles for refinement
    #     # initialize the refinement list by ±5 degrees
    #     refine_ang_list = []
    #     for result_mrc in sorted_top_n:
    #         ang = result_mrc["angle"]
    #         ang_list = np.array(
    #             np.meshgrid(
    #                 [ang[0] - 5, ang[0], ang[0] + 5],
    #                 [ang[1] - 5, ang[1], ang[1] + 5],
    #                 [ang[2] - 5, ang[2], ang[2] + 5],
    #             )
    #         ).T.reshape(-1, 3)
    #
    #         # sanity check
    #         ang_list = ang_list[(ang_list[:, 0] < 360) &
    #                             (ang_list[:, 1] < 360) &
    #                             (ang_list[:, 2] <= 180)]
    #
    #         ang_list[ang_list < 0] += 360
    #
    #         refine_ang_list.append(ang_list)
    #
    #     refine_ang_list = np.concatenate(refine_ang_list, axis=0)
    #
    #     for angle in tqdm(refine_ang_list, desc="Refining Rotation"):
    #         vec_score, vec_trans, new_vec, new_data = rot_and_search_fft(mrc_search.data,
    #                                                                      mrc_search.vec,
    #                                                                      angle,
    #                                                                      target_list,
    #                                                                      mrc_target,
    #                                                                      (a, b, c),
    #                                                                      fft_object,
    #                                                                      ifft_object,
    #                                                                      mode=mode)
    #
    #         refined_score.append({"angle": tuple(angle),
    #                               "vec_score": vec_score * rd3,
    #                               "vec_trans": vec_trans,
    #                               "vec": new_vec,
    #                               "data": new_data})
    #
    #     # sort the list to find the TopN with best scores
    #     refined_list = sorted(refined_score, key=lambda x: x["vec_score"], reverse=True)[:TopN]
    #
    # else:
    #     # no action taken when refinement is disabled
    #     refined_list = sorted_top_n

    # Write result to PDB files
    if showPDB:
        if folder is not None:
            folder_path = folder
        else:
            folder_path = Path.cwd() / "outputs" / ("VESPER_RUN_" + datetime.now().strftime('%m%d_%H%M%S'))
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(folder_path / "VEC", exist_ok=True)
        if input_pdb:
            os.makedirs(folder_path / "PDB", exist_ok=True)
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

        print(f"Voxel Trans:{result_mrc['vec_trans']}, Normalized Score: {(result_mrc['vec_score'] - ave) / std}")

        if showPDB:
            save_pdb(mrc_target.orig,
                     result_mrc["vec"],
                     result_mrc["data"],
                     sco_arr,
                     # result_mrc["vec_score"],
                     (result_mrc['vec_score'] - ave) / std,  # use normalized score
                     mrc_search.xwidth,
                     result_mrc["vec_trans"],
                     result_mrc["angle"],
                     folder_path / "VEC",
                     i)
        if input_pdb:
            rot_mtx = R.from_euler('xyz', result_mrc["angle"], degrees=True).as_matrix()
            angle_str = f"rx{int(result_mrc['angle'][0])}_ry{int(result_mrc['angle'][1])}_rz{int(result_mrc['angle'][2])}"
            trans_str = f"tx{result_mrc['vec_trans'][0]}_ty{result_mrc['vec_trans'][1]}_tz{result_mrc['vec_trans'][2]}"
            file_name = f"#{i}_{angle_str}_{trans_str}.pdb"
            save_rotated_pdb(input_pdb, rot_mtx, result_mrc["vec_trans"], str(folder_path / "PDB" / file_name))

    return refined_list


def search_map_fft_prob(mrc_target, mrc_input, mrc_P1, mrc_P2, mrc_P3, mrc_P4, mrc_search_p1, mrc_search_p2,
                        mrc_search_p3, mrc_search_p4, ang, alpha=0.0, TopN=10, vave=-1, vstd=-1, pave=-1, pstd=-1,
                        showPDB=False, folder=None, gpu=False, gpu_id=-1):
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
        mrc_input (MrcObj): the input query map
        TopN (int, optional): the number of top superimposition to find. Defaults to 10.
        ang (int, optional): search interval for angular rotation. Defaults to 30.
        vave, vstd, pave, pstd (float, optional): the average and standard deviation of
        the DOT score and probability score if known prior to search.

    Returns:
        refined_list (list): a list of refined search results including the probability score
    """

    if gpu:
        # set up torch cuda device
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. Please check your CUDA installation.")
            exit(1)
        else:
            # set up torch cuda device
            device = torch.device(f"cuda:{gpu_id}")

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

    # fftw plans initialization
    a = pyfftw.empty_aligned(mrc_search_p1.data.shape, dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned(mrc_search_p1.data.shape, dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    print()
    print("###Start Searching###")

    angle_score = []

    # init rotation grid
    search_pos_grid = np.mgrid[0:mrc_input.data.shape[0], 0:mrc_input.data.shape[0], 0:mrc_input.data.shape[0]].reshape(
        3, -1).T

    if vave >= 0 and vstd >= 0 and pstd >= 0 and pave >= 0:
        pass
    else:
        for angle in tqdm(angle_comb):
            vec_score, vec_trans, prob_score, prob_trans, _, _ = rot_and_search_fft_prob(mrc_input.data,
                                                                                         mrc_input.vec,
                                                                                         mrc_search_p1.data,
                                                                                         mrc_search_p2.data,
                                                                                         mrc_search_p3.data,
                                                                                         mrc_search_p4.data,
                                                                                         angle,
                                                                                         target_list,
                                                                                         0.0,
                                                                                         (a, b, c),
                                                                                         fft_object, ifft_object,
                                                                                         new_pos_grid=search_pos_grid)
            angle_score.append({
                "angle": angle,
                "vec_score": vec_score * rd3,
                "vec_trans": vec_trans,
                "prob_score": prob_score * rd3,
                "prob_trans": prob_trans
            })

        # calculate the ave and std for all the rotations
        score_arr_vec = np.array([row["vec_score"] for row in angle_score])
        score_arr_prob = np.array([row["prob_score"] for row in angle_score])

        vave = np.mean(score_arr_vec / rd3)
        vstd = np.std(score_arr_vec / rd3)

        pave = np.mean(score_arr_prob / rd3)
        pstd = np.std(score_arr_prob / rd3)

    print()
    print("### Result Statistics ###")
    print("Number of voxels:", mrc_target.xdim ** 3, "voxels")
    print("DotScore Std=", vstd, "DotScore Ave=", vave)
    print("ProbScore Std=", pstd, "ProbScore Ave=", pave)

    angle_score = []

    for angle in tqdm(angle_comb):
        vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans = rot_and_search_fft_prob(
            mrc_input.data,
            mrc_input.vec,
            mrc_search_p1.data,
            mrc_search_p2.data,
            mrc_search_p3.data,
            mrc_search_p4.data,
            angle,
            target_list,
            alpha,
            (a, b, c), fft_object, ifft_object,
            vstd=vstd, vave=vave, pstd=pstd, pave=pave, new_pos_grid=search_pos_grid)

        if mixed_score is None:
            mixed_score = 0
        if mixed_trans is None:
            mixed_trans = []

        norm_vec_score = (vec_score - vave) / vstd
        norm_prob_score = (prob_score - pave) / pstd

        angle_score.append({
            "angle": tuple(angle),
            "vec_score": vec_score * rd3,
            "vec_trans": vec_trans,
            "prob_score": prob_score * rd3,
            "prob_trans": prob_trans,
            "mixed_score": mixed_score,
            "mixed_trans": mixed_trans,
            "norm_vec_score": norm_vec_score,
            "norm_prob_score": norm_prob_score,
        })

    # sort the list and save topN
    sorted_score = sorted(angle_score, key=lambda x: x["mixed_score"], reverse=True)
    sorted_top_n = sorted_score[:TopN]

    # calculate mixed score statistics
    mixed_score_list = [row["mixed_score"] for row in angle_score]
    mixed_score_list = np.array(mixed_score_list)
    mixed_score_std = np.std(mixed_score_list)
    mixed_score_ave = np.mean(mixed_score_list)

    # print statistics
    print(f"MixedScore Std={mixed_score_std}, " +
          f"MixedScore Ave={mixed_score_ave}, " +
          f"Normalized by {mrc_input.xdim ** 3} voxels, " +
          f"Normalized MixedScore Ave={mixed_score_ave * rd3}, " +
          f"Normalized MixedScore Std={mixed_score_std * rd3}")

    print("### Fitted Positions ###")
    # print score list
    for result in sorted_score:
        print(format_score_result(result, mixed_score_ave, mixed_score_std))

    print()
    print("### Top", TopN, "Results ###")

    # print TopN statistics
    for idx, x in enumerate(sorted_top_n):
        print("M", str(idx + 1), format_score_result(x, mixed_score_ave, mixed_score_std))

    # 5 degrees local refinement search
    if ang > 5.0:
        print("\n###Start Refining###")
        refined_list = []
        for result_mrc in sorted_top_n:
            refined_score = []
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
                                (ang_list[:, 2] <= 180)]

            # make sure the angles are in the range of 0-360
            ang_list[ang_list < 0] += 360

            for ang in tqdm(ang_list, desc="Local Refining"):
                vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans, r_vec, r_data = rot_and_search_fft_prob(
                    mrc_input.data,
                    mrc_input.vec,
                    mrc_search_p1.data,
                    mrc_search_p2.data,
                    mrc_search_p3.data,
                    mrc_search_p4.data,
                    ang,
                    target_list,
                    alpha,
                    (a, b, c), fft_object, ifft_object,
                    ret_data=True,
                    vstd=vstd, vave=vave, pstd=pstd, pave=pave, new_pos_grid=search_pos_grid)

                refined_score.append(
                    {"angle": tuple(ang),
                     "vec_score": vec_score * rd3,
                     "vec_trans": vec_trans,
                     "vec": r_vec,
                     "data": r_data,
                     "prob_score": prob_score * rd3,
                     "prob_trans": prob_trans,
                     "mixed_score": mixed_score,
                     "mixed_trans": mixed_trans,
                     "norm_vec_score": (vec_score - vave) / vstd,
                     "norm_prob_score": (prob_score - pave) / pstd,
                     })

            refined_list.append(max(refined_score, key=lambda x: x["mixed_score"]))
        refined_list = sorted(refined_list, key=lambda x: x["mixed_score"], reverse=True)  # re-sort the list
    else:
        refined_list = sorted_top_n

    print("### Refined Fitted Positions ###")

    center_list = []
    for result in refined_list:
        center_list.append(center_of_mass(result["data"]) + result["mixed_trans"])

    clustering = DBSCAN(eps=5, min_samples=2).fit(center_list)

    # for each cluster, find the best result
    refined_list_cluster = []
    for cluster in np.unique(clustering.labels_):
        # if cluster == -1:
        #     continue
        cluster_list = [refined_list[i] for i in np.where(clustering.labels_ == cluster)[0]]
        refined_list_cluster.append(max(cluster_list, key=lambda x: x["mixed_score"]))

    # print score list
    for idx, result in enumerate(refined_list_cluster):
        print("C", str(idx + 1), format_score_result(result, mixed_score_ave, mixed_score_std))

    if showPDB:
        # Write result to PDB files
        if folder is not None:
            folder_path = Path.cwd() / folder
        else:
            folder_path = Path.cwd() / ("VESPER_RUN_" + datetime.now().strftime('%m%d_%H%M%S'))
        Path.mkdir(folder_path)
        print()
        print("###Writing results to PDB files###")
    else:
        print()

    for i, result_mrc in enumerate(refined_list):
        # convert the translation back to the original coordinate system
        r = R.from_euler('xyz', result_mrc["angle"], degrees=True)
        real_trans = convert_trans(mrc_target.cent,
                                   mrc_input.cent,
                                   r,
                                   result_mrc["mixed_trans"],
                                   mrc_input.xwidth,
                                   mrc_input.xdim)

        print("\n#" + str(i),
              "Rotation=",
              "(" + str(result_mrc["angle"][0]),
              str(result_mrc["angle"][1]),
              str(result_mrc["angle"][2]) + ")",
              "Translation=",
              "(" + "{:.3f}".format(real_trans[0]),
              "{:.3f}".format(real_trans[1]),
              "{:.3f}".format(real_trans[2]) + ")"
              )

        print("DOT Translation=", str(result_mrc["vec_trans"]),
              "Probability Score=", str(result_mrc["prob_score"]),
              "Probability Translation=", str(result_mrc["prob_trans"]))

        print("Mixed Score=", str(result_mrc["mixed_score"]), "Normalized Mix Score:",
              (result_mrc["mixed_score"] - mixed_score_ave) / mixed_score_std)

        sco_arr = get_score(
            mrc_target,
            result_mrc["data"],
            result_mrc["vec"],
            result_mrc["vec_trans"]
        )

        if showPDB:
            save_pdb(mrc_target.orig,
                     result_mrc["vec"],
                     result_mrc["data"],
                     sco_arr,
                     result_mrc["mixed_score"],
                     mrc_input.xwidth,
                     result_mrc["mixed_trans"],
                     result_mrc["angle"],
                     folder_path,
                     i)

    for i, result_mrc in enumerate(refined_list_cluster):
        if showPDB:
            save_pdb(mrc_target.orig,
                     result_mrc["vec"],
                     result_mrc["data"],
                     sco_arr,
                     result_mrc["mixed_score"],
                     mrc_input.xwidth,
                     result_mrc["mixed_trans"],
                     result_mrc["angle"],
                     folder_path,
                     i,
                     cluster=True)

    return refined_list


def rot_and_search_fft(data, vec, angle, target_list, mrc_target,
                       fft_list, fft_obj, ifft_obj,
                       mode="VecProduct", new_pos_grid=None):
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
    rot_mtx = R.from_euler("xyz", angle, degrees=True).as_matrix()

    # Rotate the query map and vector representation
    # new_vec, new_data = rot_mrc(data, vec, rot_mtx)
    new_vec, new_data = new_rot_mrc(data, vec, rot_mtx, new_pos_grid)

    if mode == "VecProduct":

        # Compose the query FFT list

        x2 = new_vec[..., 0]
        y2 = new_vec[..., 1]
        z2 = new_vec[..., 2]

        query_list_vec = [x2, y2, z2]

        # Search for best translation using FFT
        fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec, *fft_list, fft_obj, ifft_obj)

        vec_score, vec_trans = find_best_trans_list(fft_result_list_vec)

    # otherwise just use single dimension mode
    else:

        vec_score, vec_trans = fft_get_score_trans_other(target_list[0],
                                                         new_data,
                                                         fft_list[0], fft_list[1],
                                                         fft_obj, ifft_obj, mode,
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
                            fft_list, fft_obj, ifft_obj,
                            ret_data=False,
                            new_pos_grid=None,
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

    rot_mtx = R.from_euler("xyz", angle, degrees=True).as_matrix()

    # Rotate the query map and vector representation
    # r_vec, r_data, rp1, rp2, rp3, rp4 = \
    #     rot_mrc_prob(data, vec, dp1, dp2, dp3, dp4, rot_mtx)
    r_vec, r_data, rp1, rp2, rp3, rp4 = new_rot_mrc_prob(data, vec, dp1, dp2, dp3, dp4, rot_mtx, new_pos_grid)

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

    # Search for best translation using FFT
    fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec,
                                              *fft_list, fft_obj, ifft_obj)
    fft_result_list_prob = fft_search_best_dot(target_list[3:], query_list_prob,
                                               *fft_list, fft_obj, ifft_obj)

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


def eval_score_orig(mrc_target, mrc_search, angle, trans, dot_score_ave, dot_score_std):
    trans = [int(i) for i in trans]

    # Function to evaluate the DOT score for input rotation angle and translation

    # init rotation grid
    search_pos_grid = np.mgrid[0:mrc_search.data.shape[0], 0:mrc_search.data.shape[0],
                      0:mrc_search.data.shape[0]].reshape(3, -1).T

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


def eval_score_mix(mrc_target, mrc_input,
                   mrc_P1, mrc_P2, mrc_P3, mrc_P4,
                   mrc_search_p1, mrc_search_p2, mrc_search_p3, mrc_search_p4,
                   angle_list, trans_list,
                   vstd, vave, pstd, pave,
                   mix_score_ave, mix_score_std, alpha):
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

    # fftw plans initialization
    a = pyfftw.empty_aligned(mrc_search_p1.data.shape, dtype="float32")
    b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
    c = pyfftw.empty_aligned(mrc_search_p1.data.shape, dtype="float32")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
    ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    # init rotation grid
    search_pos_grid = np.mgrid[0:mrc_input.data.shape[0], 0:mrc_input.data.shape[0], 0:mrc_input.data.shape[0]].reshape(
        3, -1).T

    mix_score_list = []

    for angle, trans in zip(angle_list, trans_list):
        trans = [int(i) for i in trans]

        rot_mtx = R.from_euler("xyz", angle, degrees=True).as_matrix()

        r_vec, r_data, rp1, rp2, rp3, rp4 = new_rot_mrc_prob(mrc_input.data, mrc_input.vec,
                                                             mrc_search_p1.data,
                                                             mrc_search_p2.data,
                                                             mrc_search_p3.data,
                                                             mrc_search_p4.data,
                                                             rot_mtx, search_pos_grid)

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
