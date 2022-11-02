import os
from tqdm import tqdm
import pathlib
import pandas as pd


def contains_number(s):
    return any(i.isdigit() for i in s)


def assign_ss(pdb_path, output_dir):
    for pdb_file in tqdm(pathlib.Path(pdb_path).iterdir(), total=len(list(pathlib.Path(pdb_path).iterdir()))):
        os.system(
            "stride \"" + pdb_path + pdb_file.stem + ".pdb\" > " + "\"" + output_dir + pdb_file.stem + ".ss" + "\"")


def split_pdb_by_ss(pdb_path, ss_path):

    output_dir = "./pdb_ss_split/"

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

    with open(output_dir + pdb_path.stem + "_ssA.pdb", 'w') as fp:
        for item in a_strs['str'].to_list():
            fp.write("%s" % item)

    with open(output_dir + pdb_path.stem + "_ssB.pdb", 'w') as fp:
        for item in b_strs['str'].to_list():
            fp.write("%s" % item)

    with open(output_dir + pdb_path.stem + "_ssC.pdb", 'w') as fp:
        for item in c_strs['str'].to_list():
            fp.write("%s" % item)


def chimera_gen_mrc(map_path, mrc_path, output_map, sample_res):
    sample_res = str(sample_res)
    map_path = os.path.abspath(map_path)
    mrc_path = os.path.abspath(mrc_path)
    output_map = os.path.abspath(output_map)
    with open("./cmd_file.py", "w") as cmd_file:
        cmd_file.write('from chimera import runCommand as rc\n\n')
        cmd_file.write('rc("open ' + map_path + '")\n')
        cmd_file.write('rc("open ' + mrc_path + '")\n')
        cmd_file.write('rc("molmap #0 ' + sample_res + ' gridSpacing 1 onGrid #1' + '")\n')
        cmd_file.write('rc("vol #2 save ' + output_map + '")\n')

    run_command = 'chimera --silent --nogui ' + "./cmd_file.py 2> /dev/null"
    os.system(run_command)