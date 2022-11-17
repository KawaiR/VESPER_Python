import argparse
import os
import pathlib

import mrcfile
import numpy as np
import pandas as pd


def contains_number(s):
    return any(i.isdigit() for i in s)


def assign_ss(pdb_path, output_dir):
    print("Assigning secondary structure using Stride...")

    os.makedirs(output_dir, exist_ok=True)

    pdb_path = os.path.abspath(pdb_path)
    filename = pathlib.Path(pdb_path).stem
    os.system("stride \"" + pdb_path + "\" > " + "\"" + output_dir + filename + ".ss" + "\"")

    print("SS assignment file save to: " + output_dir + filename + ".ss")

    return output_dir + filename + ".ss"


def split_pdb_by_ss(pdb_path, ss_path, output_dir):
    print("Splitting PDB file by secondary structure...")

    os.makedirs(output_dir, exist_ok=True)

    file_ss = open(ss_path, mode="r")
    file_pdb = open(pdb_path, mode="r")

    ss_lines = file_ss.readlines()
    pdb_lines = file_pdb.readlines()

    ss_lines_pred_a = []
    ss_lines_pred_b = []
    ss_lines_pred_c = []

    for line in ss_lines:
        if line.startswith("ASG"):
            entries = line.split()
            if entries[5] == 'H' or entries[5] == 'G' or entries[5] == 'I':
                ss_lines_pred_a.append(entries[3])
            elif entries[5] == 'B' or entries[5] == 'E':
                ss_lines_pred_b.append(entries[3])
            else:
                ss_lines_pred_c.append(entries[3])

    pdb_lines_atoms = []
    residual_nums = []
    for line in pdb_lines:
        if line.startswith("ATOM"):
            pdb_lines_atoms.append(line)
            if not contains_number(line.split()[4]):
                residual_nums.append(line.split()[5])
            else:
                residual_nums.append(''.join(filter(str.isdigit, line.split()[4])))

    data_list = pd.Series(pdb_lines_atoms)
    data_res = pd.Series(residual_nums, dtype=str)
    df = pd.concat((data_list, data_res), axis=1)
    df.columns = ["str", "res_num"]

    # Create a new directory is not exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    a_strs = df[df["res_num"].isin(ss_lines_pred_a)]
    b_strs = df[df["res_num"].isin(ss_lines_pred_b)]
    c_strs = df[df["res_num"].isin(ss_lines_pred_c)]

    pdb_path = pathlib.Path(pdb_path)

    with open(output_dir + pdb_path.stem + "_ssA.pdb", 'w') as fp:
        for item in a_strs['str'].to_list():
            fp.write("%s" % item)

    with open(output_dir + pdb_path.stem + "_ssB.pdb", 'w') as fp:
        for item in b_strs['str'].to_list():
            fp.write("%s" % item)

    with open(output_dir + pdb_path.stem + "_ssC.pdb", 'w') as fp:
        for item in c_strs['str'].to_list():
            fp.write("%s" % item)

    print("PDB files saved to: " + output_dir)


def chimera_gen_mrc(pdb_path, mrc_path, output_path, sample_res):
    print("Using Chimera to generate MRC file...")

    sample_res = str(sample_res)
    pdb_path = os.path.abspath(pdb_path)
    mrc_path = os.path.abspath(mrc_path)
    output_path = os.path.abspath(output_path)

    with open("./cmd_file.py", "w") as cmd_file:
        cmd_file.write('from chimera import runCommand as rc\n\n')
        cmd_file.write('rc("open ' + pdb_path + '")\n')
        cmd_file.write('rc("open ' + mrc_path + '")\n')
        cmd_file.write('rc("molmap #0 ' + sample_res + ' gridSpacing 1 onGrid #1' + '")\n')
        cmd_file.write('rc("vol #2 save ' + output_path + '")\n')

    run_command = 'chimera --silent --nogui ' + "./cmd_file.py 2> /dev/null"
    os.system(run_command)
    # remove the cmd_file.py
    os.remove("./cmd_file.py")

    print("MRC file saved to: " + output_path)


def gen_npy(pdb_path, target_mrc, sample_res, save_npy=False):
    print("Combining MRC files into a Numpy array...")

    # remove tmp directory if exists
    if os.path.exists("./tmp_data"):
        os.system("rm -r ./tmp_data")
    #os.system("rm -r ./tmp_data")

    os.makedirs("./tmp_data/ss/", exist_ok=True)
    os.makedirs("./tmp_data/pdb/", exist_ok=True)
    os.makedirs("./tmp_data/simu_mrc/", exist_ok=True)

    out_ss = assign_ss(pdb_path, "./tmp_data/ss/")
    split_pdb_by_ss(pdb_path, out_ss, "./tmp_data/pdb/")

    for file in os.listdir("./tmp_data/pdb/"):
        if file.endswith("ssA.pdb"):
            chimera_gen_mrc("./tmp_data/pdb/" + file, target_mrc, "./tmp_data/simu_mrc/" + file + ".mrc", sample_res)
        elif file.endswith("ssB.pdb"):
            chimera_gen_mrc("./tmp_data/pdb/" + file, target_mrc, "./tmp_data/simu_mrc/" + file + ".mrc", sample_res)
        elif file.endswith("ssC.pdb"):
            chimera_gen_mrc("./tmp_data/pdb/" + file, target_mrc, "./tmp_data/simu_mrc/" + file + ".mrc", sample_res)

    with mrcfile.open(target_mrc) as mrc:
        dims = mrc.data.shape

        arr = np.zeros((4, dims[0], dims[1], dims[2]))

        for file in os.listdir("./tmp_data/simu_mrc/"):
            if file.endswith("ssC.pdb.mrc"):
                arr[0] = mrcfile.open("./tmp_data/simu_mrc/" + file).data.copy()
            elif file.endswith("ssB.pdb.mrc"):
                arr[1] = mrcfile.open("./tmp_data/simu_mrc/" + file).data.copy()
            elif file.endswith("ssA.pdb.mrc"):
                arr[2] = mrcfile.open("./tmp_data/simu_mrc/" + file).data.copy()

    arr = np.transpose(arr, (1, 2, 3, 0))

    print("Numpy array generated.")

    # get stem of pdb file
    pdb_path = pathlib.Path(pdb_path)
    pdb_stem = pdb_path.stem

    if save_npy:
        os.makedirs("./ss_npy/", exist_ok=True)
        np.save("./ss_npy/" + pdb_stem + "_prob.npy", arr)
        print("Numpy array saved to: " + "./tmp_data/npy/" + pdb_stem + "_prob.npy")

    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", type=str, help="Path to the input pdb file")
    parser.add_argument("--mrc", type=str, help="Path to the target mrc file")
    parser.add_argument("--res", type=float, help="Sampling resolution")

    # optional output path
    parser.add_argument("--save", type=bool, default=False, help="Choose to save the npy file", required=False)

    args = parser.parse_args()

    npy = gen_npy(args.pdb, args.mrc, args.res, args.save)

    print(npy.shape)
    # print stats
    print("Number in SS class Coil: ", np.count_nonzero(npy[..., 0]))
    print("Number in SS class Beta: ", np.count_nonzero(npy[..., 1]))
    print("Number in SS class Alpha: ", np.count_nonzero(npy[..., 2]))

    # print min, max, mean, std
    print("Coil: ", np.min(npy[..., 0]), np.max(npy[..., 0]), np.mean(npy[..., 0]), np.std(npy[..., 0]))
    print("Beta: ", np.min(npy[..., 1]), np.max(npy[..., 1]), np.mean(npy[..., 1]), np.std(npy[..., 1]))
    print("Alpha: ", np.min(npy[..., 2]), np.max(npy[..., 2]), np.mean(npy[..., 2]), np.std(npy[..., 2]))
