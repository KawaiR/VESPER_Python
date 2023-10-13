import os
import pathlib
import shutil

import mrcfile
import numpy as np
from TEMPy.maps.map_parser import MapParser
from TEMPy.protein.structure_blurrer import StructureBlurrer
from TEMPy.protein.structure_parser import PDBParser, mmCIFParser

import biotite.structure.io as strucio
import biotite.structure as struc


def contains_number(s):
    return any(i.isdigit() for i in s)


def split_pdb_by_ss(pdb_path, output_dir):

    array = strucio.load_structure(pdb_path)
    residues = struc.get_residues(array)[0]
    sse = struc.annotate_sse(array, chain_id="A0")

    # get res ids for each ss class
    a_res = residues[sse == "a"]
    b_res = residues[sse == "b"]
    c_res = residues[sse == "c"]

    # create ss mask by residue id
    a_mask = [(True if res_id in a_res else False) for res_id in array.res_id]
    b_mask = [(True if res_id in b_res else False) for res_id in array.res_id]
    c_mask = [(True if res_id in c_res else False) for res_id in array.res_id]

    # apply mask to array
    arr_a = array[a_mask]
    arr_b = array[b_mask]
    arr_c = array[c_mask]

    os.makedirs(output_dir, exist_ok=True)
    pdb_path = pathlib.Path(pdb_path)

    from Bio.PDB import MMCIFParser, MMCIFIO, Select

    class ResIDSelect(Select):
        def __init__(self, res_ids):
            self.res_ids = res_ids

        def accept_residue(self, residue):
            return residue.get_id()[1] in self.res_ids

    parser = MMCIFParser(QUIET=True)
    io = MMCIFIO()
    bio_st = parser.get_structure("target_pdb", pdb_path)
    io.set_structure(bio_st)
    io.save(os.path.join(output_dir, f"{pdb_path.stem}_ssA.cif"), ResIDSelect(a_res))
    io.save(os.path.join(output_dir, f"{pdb_path.stem}_ssB.cif"), ResIDSelect(b_res))
    io.save(os.path.join(output_dir, f"{pdb_path.stem}_ssC.cif"), ResIDSelect(c_res))

    # strucio.save_structure(os.path.join(output_dir, f"{pdb_path.stem}_ssA.cif"), arr_a)
    # strucio.save_structure(os.path.join(output_dir, f"{pdb_path.stem}_ssB.cif"), arr_b)
    # strucio.save_structure(os.path.join(output_dir, f"{pdb_path.stem}_ssC.cif"), arr_c)

    import gemmi
    from gemmi import cif

    block_in = cif.read(str(pdb_path)).sole_block()
    gemmi_st = gemmi.make_structure_from_block(block_in)

    block_a = cif.read(os.path.join(output_dir, f"{pdb_path.stem}_ssA.cif")).sole_block()
    block_b = cif.read(os.path.join(output_dir, f"{pdb_path.stem}_ssB.cif")).sole_block()
    block_c = cif.read(os.path.join(output_dir, f"{pdb_path.stem}_ssC.cif")).sole_block()

    gemmi_st_a = gemmi.make_structure_from_block(block_a)
    gemmi_st_a.entities = gemmi_st.entities
    gemmi_st_b = gemmi.make_structure_from_block(block_b)
    gemmi_st_b.entities = gemmi_st.entities
    gemmi_st_c = gemmi.make_structure_from_block(block_c)
    gemmi_st_c.entities = gemmi_st.entities

    gemmi_st_a.update_mmcif_block(block_a)
    gemmi_st_b.update_mmcif_block(block_b)
    gemmi_st_c.update_mmcif_block(block_c)

    block_a.write_file(os.path.join(output_dir, f"{pdb_path.stem}_ssA.cif"))
    block_b.write_file(os.path.join(output_dir, f"{pdb_path.stem}_ssB.cif"))
    block_c.write_file(os.path.join(output_dir, f"{pdb_path.stem}_ssC.cif"))


def gen_simu_map(file_path, res, output_path, densMap=None):
    """
    The gen_simu_map function takes a PDB file and generates a simulated map from it.

    :param file_path: Specify the path to the pdb file
    :param res: Set the resolution of the simulated map
    :param output_path: Specify the path to where the simulated map will be saved
    :param densMap: Specify a density map to use as a reference for output dimensions
    :return: A simulated map based on a pdb file
    """

    # check number of atoms in pdb file, if none, return new map with dimensions of densMap
    from Bio.PDB import MMCIF2Dict

    cif_dict = MMCIF2Dict.MMCIF2Dict(file_path)

    if "_atom_site.label_atom_id" in cif_dict:
        atom_site_data = cif_dict["_atom_site.label_atom_id"]
        atom_count = len(atom_site_data)
    elif "_atom_site.auth_atom_id" in cif_dict:
        atom_site_data = cif_dict["_atom_site.auth_atom_id"]
        atom_count = len(atom_site_data)
    else:
        atom_count = 0

    if atom_count == 0:
        if densMap:
            with mrcfile.open(densMap, permissive=True) as mrc:
                # set all values to 0
                with mrcfile.new(output_path, overwrite=True) as mrc_new:
                    mrc_new.set_data(np.zeros(mrc.data.shape, dtype=np.float32))
                    mrc_new.voxel_size = mrc.voxel_size
                    mrc_new.update_header_from_data()
                    mrc_new.header.nxstart = mrc.header.nxstart
                    mrc_new.header.nystart = mrc.header.nystart
                    mrc_new.header.nzstart = mrc.header.nzstart
                    mrc_new.header.origin = mrc.header.origin
                    mrc_new.header.mapc = mrc.header.mapc
                    mrc_new.header.mapr = mrc.header.mapr
                    mrc_new.header.maps = mrc.header.maps
                    mrc_new.update_header_stats()
                    mrc_new.flush() # write to disk
        else:
            raise Exception("No atoms in PDB file and no density map specified.")
    else:
        densMap = MapParser.readMRC(densMap) if densMap else None
        sb = StructureBlurrer()
        pdb_path = os.path.abspath(file_path)
        # output_path = os.path.abspath(output_path)
        if file_path.split(".")[-1] == "cif":
            st = mmCIFParser.read_mmCIF_file(pdb_path, hetatm=True)
        elif file_path.split(".")[-1] == "pdb":
            st = PDBParser.read_PDB_file("pdb1", pdb_path)
        else:
            raise ValueError("Make sure the input file is a PDB or mmCIF file.")
        simu_map = sb.gaussian_blur_real_space(st, res, densMap=densMap)
        simu_map.write_to_MRC_file(output_path)


def gen_npy(pdb_path, sample_res, npy_path=None, verbose=False):
    if verbose:
        print("Combining MRC files into a Numpy array...")

    # get stem of pdb file
    pdb_stem = str(pathlib.Path(pdb_path).stem)
    tmp_dir = f"./tmp_data/{pdb_stem}/"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    pdb_dir = os.path.join(tmp_dir, "pdb")
    simu_mrc_dir = os.path.join(tmp_dir, "simu_mrc")

    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(simu_mrc_dir, exist_ok=True)

    split_pdb_by_ss(pdb_path, pdb_dir)

    pdb_simu_map_path = os.path.join(tmp_dir, f"{pdb_stem}_simu_map.mrc")

    gen_simu_map(pdb_path, sample_res, pdb_simu_map_path, densMap=None)

    for file in os.listdir(pdb_dir):
        filename = str(pathlib.Path(file).stem)
        if "ssA" in file or "ssB" in file or "ssC" in file:
            gen_simu_map(
                os.path.join(pdb_dir, file),
                sample_res,
                os.path.join(simu_mrc_dir, filename + ".mrc"),
                densMap=pdb_simu_map_path,
            )

    with mrcfile.open(pdb_simu_map_path) as mrc:
        dims = mrc.data.shape

        arr = np.zeros((4, dims[0], dims[1], dims[2]))

        for file in os.listdir(simu_mrc_dir):
            if "ssC" in file:
                arr[0] = mrcfile.open(os.path.join(simu_mrc_dir, file)).data.copy()
            elif "ssB" in file:
                arr[1] = mrcfile.open(os.path.join(simu_mrc_dir, file)).data.copy()
            elif "ssA" in file:
                arr[2] = mrcfile.open(os.path.join(simu_mrc_dir, file)).data.copy()

        arr = np.transpose(arr, (1, 2, 3, 0))

    # get stem of pdb file
    pdb_path = pathlib.Path(pdb_path)
    pdb_stem = str(pdb_path.stem)

    if npy_path:
        os.makedirs(npy_path, exist_ok=True)
        save_pth = os.path.join(npy_path, pdb_stem + "_prob.npy")
        np.save(save_pth, arr)
        print("Numpy array saved to: " + save_pth)

    if verbose:
        print(arr.shape)
        # print stats
        print("Number in SS class Coil: ", np.count_nonzero(arr[..., 0]))
        print("Number in SS class Beta: ", np.count_nonzero(arr[..., 1]))
        print("Number in SS class Alpha: ", np.count_nonzero(arr[..., 2]))

        # print min, max, mean, std
        print(
            "Coil: ",
            np.min(arr[..., 0]),
            np.max(arr[..., 0]),
            np.mean(arr[..., 0]),
            np.std(arr[..., 0]),
        )
        print(
            "Beta: ",
            np.min(arr[..., 1]),
            np.max(arr[..., 1]),
            np.mean(arr[..., 1]),
            np.std(arr[..., 1]),
        )
        print(
            "Alpha: ",
            np.min(arr[..., 2]),
            np.max(arr[..., 2]),
            np.mean(arr[..., 2]),
            np.std(arr[..., 2]),
        )

    return arr
