# coding: utf-8
import multiprocessing
import os
from datetime import datetime

import concurrent.futures
from itertools import product
from pathlib import Path

import mrcfile
import pyfftw
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from scipy.ndimage import center_of_mass

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBIO import PDBIO

from utils import *

pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"
pyfftw.config.NUM_THREADS = max(multiprocessing.cpu_count() - 2, 2)  # Maybe the CPU is sweating too much?


class MrcObj:
    """A mrc object that represents the density data and statistics of a given mrc file"""

    def __init__(self, path):
        # open the specified mrcfile and read the header information
        mrc = mrcfile.open(path, permissive=True)
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
        self.cent = np.array(
            [
                self.xdim * 0.5,
                self.ydim * 0.5,
                self.zdim * 0.5,
            ]
        )

        # read and store the origin coordinate from the header
        self.orig = np.array((header.origin.x, header.origin.y, header.origin.z))

        if np.all(self.orig == 0):
            # MRC2000 format uses nxstart, nystart, nzstart instead of origin
            self.orig_idx = np.array((header.nxstart, header.nystart, header.nzstart))
            self.orig = self.orig_idx * np.array((self.xwidth, self.ywidth, self.zwidth))

        # swap the xz axes of density data array and store in self.data
        # also convert the data type to float32
        self.data = np.swapaxes(copy.deepcopy(data), 0, 2).astype(np.float32)

        # create 1d representation of the density value by flattening the data array
        self.dens = data.flatten()

        # initialize the vector array to be same shape as data but will all zeros
        self.vec = np.zeros((self.xdim, self.ydim, self.zdim, 3), dtype="float32")

        # initialize all the statistics values
        self.dsum = None  # total density value
        self.Nact = None  # non-zero density voxel count
        self.ave = None  # average density value
        self.std_norm_ave = None  # L2 norm normalized with average density value
        self.std = None  # denormalize L2 norm


def fft_search_best_dot(target_list, query_list, a, b, c, fft_object, ifft_object):
    """A better version of the fft_search_score_trans function that finds the best dot product for the target and
    query list of vectors.

    Args:
        target_list (list(numpy.array)): FFT transformed result from target map (any dimensions)
        query_list (list(numpy.array)): the input query map vector array (must have the same dimensions as target_list)
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


def search_map_fft(
    mrc_ref,
    mrc_tgt,
    TopN=10,
    ang=30,
    mode="VecProduct",
    is_eval_mode=False,
    save_path=".",
    showPDB=False,
    folder=None,
    gpu=False,
    device=None,
    remove_dup=False,
    ldp_path=None,
    backbone_path=None,
    input_pdb=None,
    input_mrc=None,
    threads=2,
):
    """The main search function for fining the best superimposition for the target and the query map.

    Args:
        mrc_ref (MrcObj): the input target map
        mrc_tgt (MrcObj): the input query map
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

        _ = get_score(mrc_ref, mrc_tgt.data, mrc_tgt.vec, [0, 0, 0])

        exit(0)

    # init rotation grid
    search_pos_grid = (
        np.mgrid[
            0 : mrc_tgt.data.shape[0],
            0 : mrc_tgt.data.shape[0],
            0 : mrc_tgt.data.shape[0],
        ]
        .reshape(3, -1)
        .T
    )

    # init the target map vectors
    x1 = copy.deepcopy(mrc_ref.vec[:, :, :, 0])

    # Postprocessing for other modes
    if mode == "Overlap":
        x1 = np.where(mrc_ref.data > 0, 1.0, 0.0)
    elif mode == "CC":
        x1 = np.where(mrc_ref.data > 0, mrc_ref.data, 0.0)
    elif mode == "PCC":
        x1 = np.where(mrc_ref.data > 0, mrc_ref.data - mrc_ref.ave, 0.0)
    elif mode == "Laplacian":
        x1 = laplacian_filter(mrc_ref.data)

    rd3 = 1.0 / mrc_ref.data.size

    # init fft transformation for the target map
    X1 = np.fft.rfftn(x1)
    X1 = np.conj(X1)

    # calculate combination of rotation angles
    angle_comb = calc_angle_comb(ang)

    target_list = [X1]

    # init fft transformation for the target map
    if mode == "VecProduct":
        y1 = copy.deepcopy(mrc_ref.vec[:, :, :, 1])
        z1 = copy.deepcopy(mrc_ref.vec[:, :, :, 2])
        Y1 = np.fft.rfftn(y1)
        Y1 = np.conj(Y1)
        Z1 = np.fft.rfftn(z1)
        Z1 = np.conj(Z1)
        target_list = [X1, Y1, Z1]

    if gpu:
        # move target list to GPU
        import torch
        target_list = [torch.from_numpy(target_list[i]).to(device).share_memory_() for i in range(len(target_list))]
    else:
        # fftw plans initialization
        a = pyfftw.empty_aligned(mrc_tgt.vec[..., 0].shape, dtype="float32")
        b = pyfftw.empty_aligned((a.shape[0], a.shape[1], a.shape[2] // 2 + 1), dtype="complex64")
        c = pyfftw.empty_aligned(mrc_tgt.vec[..., 0].shape, dtype="float32")

        fft_object = pyfftw.FFTW(a, b, axes=(0, 1, 2))
        ifft_object = pyfftw.FFTW(b, c, direction="FFTW_BACKWARD", axes=(0, 1, 2), normalise_idft=False)

    print("###Start Searching###")

    angle_score = []

    # search process
    if gpu:
        # use torch multiprocessing
        import torch.multiprocessing as mp

        # move everything to GPU
        data = torch.from_numpy(mrc_tgt.data).to(device).share_memory_()
        vec = torch.from_numpy(mrc_tgt.vec).to(device).share_memory_()
        new_pos_grid = torch.from_numpy(search_pos_grid).to(device).share_memory_()

        # no multiprocessing
        # for angle in tqdm(angle_comb):
        #     vec_score, vec_trans = gpu_rot_and_search_fft(
        #         data,
        #         vec,
        #         angle,
        #         target_list,
        #         mrc_ref,
        #         device,
        #         mode,
        #         new_pos_grid,
        #         False
        #     )
        # angle_score.append(
        #     {
        #         "angle": angle,
        #         "vec_score": vec_score * rd3,
        #         "vec_trans": vec_trans,
        #         "ldp_recall": 0.0,
        #     }
        # )

        # process pool
        # mp.set_start_method("spawn")
        # with tqdm(total=len(angle_comb)) as pbar:
        #     with mp.Pool(4) as pool:
        #         results = [
        #             (
        #                 rot_ang,
        #                 pool.apply_async(
        #                     gpu_rot_and_search_fft,
        #                     args=(data, vec, rot_ang, target_list, mrc_ref, device, mode, new_pos_grid, False)
        #                 ),
        #             )
        #             for rot_ang in angle_comb
        #         ]
        #         for result in results:
        #             rot_ang, future = result
        #             result = future.get()
        #             pbar.update(1)
        #             angle_score.append(
        #                 {
        #                     "angle": rot_ang,
        #                     "vec_score": result[0] * rd3,
        #                     "vec_trans": result[1],
        #                     "ldp_recall": 0.0,
        #                 }
        #             )

        # concurrent futures thread pool
        with tqdm(total=len(angle_comb)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {
                    executor.submit(
                        gpu_rot_and_search_fft,
                        data,
                        vec,
                        rot_ang,
                        target_list,
                        mrc_ref,
                        device,
                        mode,
                        new_pos_grid,
                        False,
                    ): rot_ang
                    for rot_ang in angle_comb
                }
                for future in concurrent.futures.as_completed(futures):
                    rot_ang = futures[future]
                    result = future.result()
                    pbar.update(1)
                    angle_score.append(
                        {
                            "angle": rot_ang,
                            "vec_score": result[0] * rd3,
                            "vec_trans": result[1],
                            "ldp_recall": 0.0,
                        }
                    )

    else:
        for angle in tqdm(angle_comb):
            vec_score, vec_trans, _, _ = rot_and_search_fft(
                mrc_tgt.data,
                mrc_tgt.vec,
                angle,
                target_list,
                mrc_ref,
                (a, b, c),
                fft_object,
                ifft_object,
                mode=mode,
                new_pos_grid=search_pos_grid,
            )
            angle_score.append(
                {
                    "angle": angle,
                    "vec_score": vec_score * rd3,
                    "vec_trans": vec_trans,
                    "ldp_recall": 0.0,
                }
            )

    # calculate the ave and std
    score_arr = np.array([row["vec_score"] for row in angle_score])
    ave = np.mean(score_arr)
    std = np.std(score_arr)
    print("\nStd= " + str(std) + " Ave= " + str(ave) + "\n")

    # sort the results by score
    angle_score = sorted(angle_score, key=lambda k: k["vec_score"], reverse=True)

    # LDP Recall calculation and sort
    ldp_recall_mode = (ldp_path is not None) + (backbone_path is not None)
    if ldp_recall_mode == 1:
        print("LDP Recall mode is not complete. Please provide both ldp and backbone files.")
        ldp_recall_mode = False
    elif ldp_recall_mode == 2:
        ldp_recall_mode = True

    if ldp_recall_mode:
        # get atom coords from ldp
        ldp_atoms = []
        with open(ldp_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    ldp_atoms.append(np.array((float(line[30:38]), float(line[38:46]), float(line[46:54]))))

        assert len(ldp_atoms) > 0, "No points found in LDP file."
        ldp_atoms = torch.from_numpy(np.array(ldp_atoms)).to(device)

        # get ca atoms from backbone
        backbone_ca = []
        with open(backbone_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":  # only CA atoms
                    # if tokens[0] == "ATOM": # all atoms
                    backbone_ca.append(np.array((float(line[30:38]), float(line[38:46]), float(line[46:54]))))

        assert len(backbone_ca) > 0, "No CA atoms found in backbone file."
        backbone_ca = torch.from_numpy(np.array(backbone_ca)).to(device)

        # calculate for each rotation
        for result_item in tqdm(angle_score, desc="Calculating LDP Recall Score"):
            r = R.from_euler("xyz", result_item["angle"], degrees=True)
            rot_mtx = r.inv().as_matrix()
            rot_mtx = torch.from_numpy(rot_mtx).to(device)
            real_trans = convert_trans(
                mrc_ref.cent,
                mrc_tgt.cent,
                r,
                result_item["vec_trans"],
                mrc_tgt.xwidth,
                mrc_tgt.xdim,
            )
            result_item["ldp_recall"] = calc_ldp_recall_score(
                ldp_atoms,
                backbone_ca,
                rot_mtx,
                torch.from_numpy(np.array(real_trans)).to(device),
                device,
            )

        # sort by LDP recall
        angle_score = sorted(angle_score, key=lambda x: x["ldp_recall"], reverse=True)

    if remove_dup:
        new_angle_score = []

        print("###Start Duplicate Removal###")

        # duplicate removal
        hash_angs = {}

        non_dup_count = 0

        # at least 2 angle spacings apart
        # n_angles_apart = 2
        n_angles_apart = 30 // ang
        ang_range = n_angles_apart * int(ang)
        ang_range = int(ang_range)

        for result_mrc in tqdm(angle_score, desc="Removing Duplicates"):
            # duplicate removal
            if tuple(result_mrc["angle"]) in hash_angs.keys():
                # print(f"Duplicate: {result_mrc['angle']}")
                trans = hash_angs[tuple(result_mrc["angle"])]
                # manhattan distance
                if np.sum(np.abs(trans - result_mrc["vec_trans"])) < mrc_tgt.xdim:
                    # result_mrc["vec_score"] = 0
                    continue

            # add to hash
            hash_angs[tuple(result_mrc["angle"])] = np.array(result_mrc["vec_trans"])

            angle_x = int(result_mrc["angle"][0])
            angle_y = int(result_mrc["angle"][1])
            angle_z = int(result_mrc["angle"][2])

            # add surrounding angles to hash
            for xx in range(angle_x - ang_range, angle_x + ang_range + 1, int(ang)):
                for yy in range(angle_y - ang_range, angle_y + ang_range + 1, int(ang)):
                    for zz in range(angle_z - ang_range, angle_z + ang_range + 1, int(ang)):
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
            new_angle_score.append(result_mrc)

        print("#Non-duplicate count: " + str(non_dup_count))
        angle_score = new_angle_score

    # sort the list and get top N results
    if ldp_recall_mode:
        sorted_top_n = sorted(angle_score, key=lambda x: x["ldp_recall"], reverse=True)[:TopN]
    else:
        sorted_top_n = sorted(angle_score, key=lambda x: x["vec_score"], reverse=True)[:TopN]

    for i, result_mrc in enumerate(sorted_top_n):
        r = R.from_euler("xyz", result_mrc["angle"], degrees=True)
        new_trans = convert_trans(
            mrc_ref.cent,
            mrc_tgt.cent,
            r,
            result_mrc["vec_trans"],
            mrc_tgt.xwidth,
            mrc_tgt.xdim,
        )

        print(
            "M" + str(i),
            "Rotation=",
            "(" + str(result_mrc["angle"][0]),
            str(result_mrc["angle"][1]),
            str(result_mrc["angle"][2]) + ")",
            "Translation=",
            "(" + "{:.3f}".format(new_trans[0]),
            "{:.3f}".format(new_trans[1]),
            "{:.3f}".format(new_trans[2]) + ")",
        )

        if ldp_recall_mode:
            print(f"LDP Recall Score: {result_mrc['ldp_recall']}")

    print()

    if ang >= 5.0:
        print("###Start Refining###")
        refined_list = []
        for result_mrc in tqdm(sorted_top_n, desc="Refining Top N", position=0):
            refined_score = []
            ang = result_mrc["angle"]

            interval = 2

            # 2 degrees interval refinement
            x_list = range(int(ang[0]) - 5, int(ang[0]) + 6, interval)
            y_list = range(int(ang[1]) - 5, int(ang[1]) + 6, interval)
            z_list = range(int(ang[2]) - 5, int(ang[2]) + 6, interval)

            ang_list = np.array(list(product(x_list, y_list, z_list))).astype(np.float32)  # convert angle to float32

            # remove duplicates
            ang_list = ang_list[(ang_list[:, 0] < 360) & (ang_list[:, 1] < 360) & (ang_list[:, 2] < 180)]

            # make sure the angles are in the range of 0-360
            ang_list[ang_list < 0] += 360
            ang_list[ang_list > 360] -= 360

            if gpu:
                data_gpu = torch.from_numpy(mrc_tgt.data).to(device).share_memory_()
                vec_gpu = torch.from_numpy(mrc_tgt.vec).to(device).share_memory_()
                new_pos_grid_gpu = torch.from_numpy(search_pos_grid).to(device).share_memory_()

                with tqdm(total=len(ang_list), position=1, leave=False) as pbar:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                        futures = {
                            executor.submit(
                                gpu_rot_and_search_fft,
                                data_gpu,
                                vec_gpu,
                                rot_ang,
                                target_list,
                                mrc_ref,
                                device,
                                mode,
                                new_pos_grid_gpu,
                                True,
                            ): rot_ang
                            for rot_ang in ang_list
                        }
                        for future in concurrent.futures.as_completed(futures):
                            rot_ang = futures[future]
                            result = future.result()
                            pbar.update(1)
                            refined_score.append(
                                {
                                    "angle": rot_ang,
                                    "vec_score": result[0] * rd3,
                                    "vec_trans": result[1],
                                    "vec": result[2],
                                    "data": result[3],
                                    "ldp_recall": 0.0,
                                }
                            )
            else:
                for angle in tqdm(ang_list, position=1, leave=False):
                    vec_score, vec_trans, new_vec, new_data = rot_and_search_fft(
                        mrc_tgt.data,
                        mrc_tgt.vec,
                        angle,
                        target_list,
                        mrc_ref,
                        (a, b, c),
                        fft_object,
                        ifft_object,
                        mode=mode,
                        new_pos_grid=search_pos_grid,
                    )
                    refined_score.append(
                        {
                            "angle": tuple(angle),
                            "vec_score": vec_score * rd3,
                            "vec_trans": vec_trans,
                            "vec": new_vec,
                            "data": new_data,
                        }
                    )
            if ldp_recall_mode:
                for item in refined_score:
                    r = R.from_euler("xyz", item["angle"], degrees=True)
                    rot_mtx = r.inv().as_matrix()
                    rot_mtx = torch.from_numpy(rot_mtx).to(device)
                    real_trans = convert_trans(
                        mrc_ref.cent,
                        mrc_tgt.cent,
                        r,
                        item["vec_trans"],
                        mrc_tgt.xwidth,
                        mrc_tgt.xdim,
                    )
                    ldp_recall = calc_ldp_recall_score(
                        ldp_atoms,
                        backbone_ca,
                        rot_mtx,
                        torch.from_numpy(np.array(real_trans)).to(device),
                        device,
                    )
                    item["ldp_recall"] = ldp_recall
                refined_list.append(max(refined_score, key=lambda x: x["ldp_recall"]))
            else:
                refined_list.append(max(refined_score, key=lambda x: x["vec_score"]))
        if ldp_recall_mode:
            refined_list = sorted(refined_list, key=lambda x: x["ldp_recall"], reverse=True)
        else:
            refined_list = sorted(refined_list, key=lambda x: x["vec_score"], reverse=True)
    else:
        # TODO: fix vec and data here
        refined_list = sorted_top_n

    # Write result to PDB files
    if showPDB or input_pdb:
        if folder is not None:
            folder_path = folder
        else:
            folder_path = Path.cwd() / "outputs" / ("VESPER_RUN_" + datetime.now().strftime("%m%d_%H%M%S"))
        os.makedirs(folder_path, exist_ok=True)
        print("\nOutput Folder:", os.path.abspath(folder_path))
        if showPDB:
            print("###Writing vector results to PDB files###")
            os.makedirs(os.path.join(folder_path, "VEC"), exist_ok=True)
        if input_pdb:
            print("###Writing transformed PDB files###")
            os.makedirs(os.path.join(folder_path, "PDB"), exist_ok=True)
        if input_mrc:
            print("###Writing transformed MRC files###")
            os.makedirs(os.path.join(folder_path, "MRC"), exist_ok=True)
    else:
        print()

    for i, result_mrc in enumerate(refined_list):
        r = R.from_euler("xyz", result_mrc["angle"], degrees=True)
        new_trans = convert_trans(
            mrc_ref.cent,
            mrc_tgt.cent,
            r,
            result_mrc["vec_trans"],
            mrc_tgt.xwidth,
            mrc_tgt.xdim,
        )

        print(
            "\n#" + str(i),
            "Rotation=",
            "(" + str(result_mrc["angle"][0]),
            str(result_mrc["angle"][1]),
            str(result_mrc["angle"][2]) + ")",
            "Translation=",
            "(" + "{:.3f}".format(new_trans[0]),
            "{:.3f}".format(new_trans[1]),
            "{:.3f}".format(new_trans[2]) + ")",
        )

        sco_arr = get_score(mrc_ref, result_mrc["data"], result_mrc["vec"], result_mrc["vec_trans"])

        print(f"Voxel Trans:{result_mrc['vec_trans']}, Normalized Score: {(result_mrc['vec_score'] - ave) / std}")

        if ldp_recall_mode:
            print(f"LDP Recall Score: {result_mrc['ldp_recall']}")

        rot_mtx = R.from_euler("xyz", result_mrc["angle"], degrees=True).inv().as_matrix()
        true_trans = convert_trans(
            mrc_ref.cent,
            mrc_tgt.cent,
            r,
            result_mrc["vec_trans"],
            mrc_tgt.xwidth,
            mrc_tgt.xdim,
        )

        angle_str = f"rx{int(result_mrc['angle'][0])}_ry{int(result_mrc['angle'][1])}_rz{int(result_mrc['angle'][2])}"
        trans_str = f"tx{true_trans[0]:.3f}_ty{true_trans[1]:.3f}_tz{true_trans[2]:.3f}"

        # output stuff
        if showPDB:
            save_vec_as_pdb(
                mrc_ref.orig,
                result_mrc["vec"],
                result_mrc["data"],
                sco_arr,
                # result_mrc["vec_score"],
                (result_mrc["vec_score"] - ave) / std,  # use normalized score
                mrc_tgt.xwidth,
                result_mrc["vec_trans"],
                result_mrc["angle"],
                os.path.join(folder_path, "VEC"),
                i,
            )
        if input_mrc:
            file_name = f"#{i}_{angle_str}_{trans_str}.mrc"
            save_rotated_mrc(input_mrc, os.path.join(folder_path, "MRC", file_name), result_mrc["angle"], true_trans)
        if input_pdb:
            file_name = f"#{i}_{angle_str}_{trans_str}.pdb"
            pdbio = PDBIO()
            # check input file format
            if input_pdb.split(".")[-1] == "pdb":
                parser = PDBParser(QUIET=True)
                save_rotated_pdb(
                    input_pdb,
                    rot_mtx,
                    true_trans,
                    os.path.join(folder_path, "PDB", file_name),
                    parser,
                    pdbio,
                )
            elif input_pdb.split(".")[-1] == "cif":
                parser = MMCIFParser(QUIET=True)
                save_rotated_pdb(
                    input_pdb,
                    rot_mtx,
                    true_trans,
                    os.path.join(folder_path, "PDB", file_name),
                    parser,
                    pdbio,
                )
            else:
                print("Input file is not pdb or cif format. No transform PDB will be generated.")

    return refined_list


def search_map_fft_prob(
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
    ang,
    alpha=0.0,
    TopN=10,
    vave=-1,
    vstd=-1,
    pave=-1,
    pstd=-1,
    showPDB=False,
    folder=None,
    gpu=False,
    gpu_id=-1,
):
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
    search_pos_grid = (
        np.mgrid[
            0 : mrc_input.data.shape[0],
            0 : mrc_input.data.shape[0],
            0 : mrc_input.data.shape[0],
        ]
        .reshape(3, -1)
        .T
    )

    if vave >= 0 and vstd >= 0 and pstd >= 0 and pave >= 0:
        pass
    else:
        for angle in tqdm(angle_comb):
            (vec_score, vec_trans, prob_score, prob_trans, _, _,) = rot_and_search_fft_prob(
                mrc_input.data,
                mrc_input.vec,
                mrc_search_p1.data,
                mrc_search_p2.data,
                mrc_search_p3.data,
                mrc_search_p4.data,
                angle,
                target_list,
                0.0,
                (a, b, c),
                fft_object,
                ifft_object,
                new_pos_grid=search_pos_grid,
            )
            angle_score.append(
                {
                    "angle": angle,
                    "vec_score": vec_score * rd3,
                    "vec_trans": vec_trans,
                    "prob_score": prob_score * rd3,
                    "prob_trans": prob_trans,
                }
            )

        # calculate the ave and std for all the rotations
        score_arr_vec = np.array([row["vec_score"] for row in angle_score])
        score_arr_prob = np.array([row["prob_score"] for row in angle_score])

        vave = np.mean(score_arr_vec / rd3)
        vstd = np.std(score_arr_vec / rd3)

        pave = np.mean(score_arr_prob / rd3)
        pstd = np.std(score_arr_prob / rd3)

    print()
    print("### Result Statistics ###")
    print("Number of voxels:", mrc_target.xdim**3, "voxels")
    print("DotScore Std=", vstd, "DotScore Ave=", vave)
    print("ProbScore Std=", pstd, "ProbScore Ave=", pave)

    angle_score = []

    for angle in tqdm(angle_comb):
        (vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans,) = rot_and_search_fft_prob(
            mrc_input.data,
            mrc_input.vec,
            mrc_search_p1.data,
            mrc_search_p2.data,
            mrc_search_p3.data,
            mrc_search_p4.data,
            angle,
            target_list,
            alpha,
            (a, b, c),
            fft_object,
            ifft_object,
            vstd=vstd,
            vave=vave,
            pstd=pstd,
            pave=pave,
            new_pos_grid=search_pos_grid,
        )

        if mixed_score is None:
            mixed_score = 0
        if mixed_trans is None:
            mixed_trans = []

        norm_vec_score = (vec_score - vave) / vstd
        norm_prob_score = (prob_score - pave) / pstd

        angle_score.append(
            {
                "angle": tuple(angle),
                "vec_score": vec_score * rd3,
                "vec_trans": vec_trans,
                "prob_score": prob_score * rd3,
                "prob_trans": prob_trans,
                "mixed_score": mixed_score,
                "mixed_trans": mixed_trans,
                "norm_vec_score": norm_vec_score,
                "norm_prob_score": norm_prob_score,
            }
        )

    # sort the list and save topN
    sorted_score = sorted(angle_score, key=lambda x: x["mixed_score"], reverse=True)
    sorted_top_n = sorted_score[:TopN]

    # calculate mixed score statistics
    mixed_score_list = [row["mixed_score"] for row in angle_score]
    mixed_score_list = np.array(mixed_score_list)
    mixed_score_std = np.std(mixed_score_list)
    mixed_score_ave = np.mean(mixed_score_list)

    # print statistics
    print(
        f"MixedScore Std={mixed_score_std}, "
        + f"MixedScore Ave={mixed_score_ave}, "
        + f"Normalized by {mrc_input.xdim ** 3} voxels, "
        + f"Normalized MixedScore Ave={mixed_score_ave * rd3}, "
        + f"Normalized MixedScore Std={mixed_score_std * rd3}"
    )

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
            ang_list = ang_list[(ang_list[:, 0] < 360) & (ang_list[:, 1] < 360) & (ang_list[:, 2] <= 180)]

            # make sure the angles are in the range of 0-360
            ang_list[ang_list < 0] += 360

            for ang in tqdm(ang_list, desc="Local Refining"):
                (
                    vec_score,
                    vec_trans,
                    prob_score,
                    prob_trans,
                    mixed_score,
                    mixed_trans,
                    r_vec,
                    r_data,
                ) = rot_and_search_fft_prob(
                    mrc_input.data,
                    mrc_input.vec,
                    mrc_search_p1.data,
                    mrc_search_p2.data,
                    mrc_search_p3.data,
                    mrc_search_p4.data,
                    ang,
                    target_list,
                    alpha,
                    (a, b, c),
                    fft_object,
                    ifft_object,
                    ret_data=True,
                    vstd=vstd,
                    vave=vave,
                    pstd=pstd,
                    pave=pave,
                    new_pos_grid=search_pos_grid,
                )

                refined_score.append(
                    {
                        "angle": tuple(ang),
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
                    }
                )

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
        print(
            "C",
            str(idx + 1),
            format_score_result(result, mixed_score_ave, mixed_score_std),
        )

    if showPDB:
        # Write result to PDB files
        if folder is not None:
            folder_path = Path.cwd() / folder
        else:
            folder_path = Path.cwd() / ("VESPER_RUN_" + datetime.now().strftime("%m%d_%H%M%S"))
        Path.mkdir(folder_path)
        print()
        print("###Writing results to PDB files###")
    else:
        print()

    for i, result_mrc in enumerate(refined_list):
        # convert the translation back to the original coordinate system
        r = R.from_euler("xyz", result_mrc["angle"], degrees=True)
        real_trans = convert_trans(
            mrc_target.cent,
            mrc_input.cent,
            r,
            result_mrc["mixed_trans"],
            mrc_input.xwidth,
            mrc_input.xdim,
        )

        print(
            "\n#" + str(i),
            "Rotation=",
            "(" + str(result_mrc["angle"][0]),
            str(result_mrc["angle"][1]),
            str(result_mrc["angle"][2]) + ")",
            "Translation=",
            "(" + "{:.3f}".format(real_trans[0]),
            "{:.3f}".format(real_trans[1]),
            "{:.3f}".format(real_trans[2]) + ")",
        )

        print(
            "DOT Translation=",
            str(result_mrc["vec_trans"]),
            "Probability Score=",
            str(result_mrc["prob_score"]),
            "Probability Translation=",
            str(result_mrc["prob_trans"]),
        )

        print(
            "Mixed Score=",
            str(result_mrc["mixed_score"]),
            "Normalized Mix Score:",
            (result_mrc["mixed_score"] - mixed_score_ave) / mixed_score_std,
        )

        sco_arr = get_score(mrc_target, result_mrc["data"], result_mrc["vec"], result_mrc["vec_trans"])

        if showPDB:
            save_vec_as_pdb(
                mrc_target.orig,
                result_mrc["vec"],
                result_mrc["data"],
                sco_arr,
                result_mrc["mixed_score"],
                mrc_input.xwidth,
                result_mrc["mixed_trans"],
                result_mrc["angle"],
                folder_path,
                i,
            )

    for i, result_mrc in enumerate(refined_list_cluster):
        if showPDB:
            save_vec_as_pdb(
                mrc_target.orig,
                result_mrc["vec"],
                result_mrc["data"],
                sco_arr,
                result_mrc["mixed_score"],
                mrc_input.xwidth,
                result_mrc["mixed_trans"],
                result_mrc["angle"],
                folder_path,
                i,
                cluster=True,
            )

    return refined_list


def rot_and_search_fft(
    data,
    vec,
    angle,
    target_list,
    mrc_target,
    fft_list,
    fft_obj,
    ifft_obj,
    mode="VecProduct",
    new_pos_grid=None,
):
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

        vec_score, vec_trans = fft_get_score_trans_other(
            target_list[0],
            new_data,
            fft_list[0],
            fft_list[1],
            fft_obj,
            ifft_obj,
            mode,
            mrc_target.ave,
        )

        if mode == "CC":
            rstd2 = 1.0 / mrc_target.std**2
            vec_score = vec_score * rstd2
        if mode == "PCC":
            rstd3 = 1.0 / mrc_target.std_norm_ave**2
            vec_score = vec_score * rstd3

    return vec_score, vec_trans, new_vec, new_data


def rot_and_search_fft_prob(
    data,
    vec,
    dp1,
    dp2,
    dp3,
    dp4,
    angle,
    target_list,
    alpha,
    fft_list,
    fft_obj,
    ifft_obj,
    ret_data=False,
    new_pos_grid=None,
    vstd=None,
    vave=None,
    pstd=None,
    pave=None,
):
    """Calculate the best translation for the query map given a rotation angle

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
    fft_result_list_vec = fft_search_best_dot(target_list[:3], query_list_vec, *fft_list, fft_obj, ifft_obj)
    fft_result_list_prob = fft_search_best_dot(target_list[3:], query_list_prob, *fft_list, fft_obj, ifft_obj)

    vec_score, vec_trans = find_best_trans_list(fft_result_list_vec)
    prob_score, prob_trans = find_best_trans_list(fft_result_list_prob)

    mixed_score, mixed_trans = None, None

    if vstd is not None and vave is not None and pstd is not None and pave is not None:
        mixed_score, mixed_trans = find_best_trans_mixed(fft_result_list_vec, fft_result_list_prob, alpha, vstd, vave, pstd, pave)

    if ret_data:
        return (
            vec_score,
            vec_trans,
            prob_score,
            prob_trans,
            mixed_score,
            mixed_trans,
            r_vec,
            r_data,
        )
    return vec_score, vec_trans, prob_score, prob_trans, mixed_score, mixed_trans
