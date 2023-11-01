import argparse
import os
import time
from enum import Enum

from fitter import MapFitter
from map import EMmap, unify_dims
import numpy as np
import sys


class Mode(Enum):
    V = "VecProduct"
    O = "Overlap"
    C = "CC"
    P = "PCC"
    L = "Laplacian"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")

    orig = subparser.add_parser("orig")
    ss = subparser.add_parser("ss")

    # original VESPER menu
    orig.add_argument("-a", type=str, required=True, help="MAP1.mrc (large)")
    orig.add_argument("-b", type=str, required=True, help="MAP2.mrc (small)")
    orig.add_argument("-t", type=float, default=0.0, help="Threshold of density map1")
    orig.add_argument("-T", type=float, default=0.0, help="Threshold of density map2")
    orig.add_argument(
        "-g", type=float, default=16.0, help="Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]"
    )
    orig.add_argument("-s", type=float, default=7.0, help="Sampling voxel spacing def=7.0")
    orig.add_argument("-A", type=float, default=30.0, help="Sampling angle spacing def=30.0")
    orig.add_argument("-N", type=int, default=10, help="Refine Top [int] models def=10")
    orig.add_argument("-S", action="store_true", default=False, help="Show topN models in PDB format def=false")
    orig.add_argument(
        "-M",
        type=str,
        default="V",
        help="V: vector product mode (default)\n"
        + "O: overlap mode\n"
        + "C: Cross Correlation Coefficient Mode\n"
        + "P: Pearson Correlation Coefficient Mode\n"
        + "L: Laplacian Filtering Mode",
    )
    orig.add_argument("-E", type=bool, default=False, help="Evaluation mode of the current position def=false")
    orig.add_argument("-o", type=str, default=None, help="Output folder name")
    orig.add_argument("-gpu", type=int, help="GPU ID to use for CUDA acceleration def=0")
    orig.add_argument("-nodup", action="store_true", default=False, help="Remove duplicate models using heuristics def=false")
    orig.add_argument("-ldp", type=str, default=None, help="Path to the local dense point file def=None")
    orig.add_argument("-ca", type=str, default=None, help="Path to the CA file def=None")
    orig.add_argument("-pdbin", type=str, default=None, help="Input PDB file to be transformed def=None")
    orig.add_argument("-mrcout", action="store_true", default=False, help="Output the transformed query map def=false")
    orig.add_argument("-c", type=int, default=2, help="Number of threads to use def=2")
    orig.add_argument("-al", type=float, default=None, help="Angle limit for searching def=None")
    orig.add_argument("-res", type=float, default=None, help="Resolution of the experimental map used to create simulated map from structure")

    # secondary structure matching menu
    ss.add_argument("-a", type=str, required=True, help="MAP1.mrc (large)")
    ss.add_argument("-b", type=str, required=True, help="MAP2.mrc (small)")
    ss.add_argument("-npb", type=str, required=False, help="numpy array for Predictions for map 2")
    ss.add_argument("-npa", type=str, required=True, help="numpy array for Predictions for map 1")
    ss.add_argument(
        "-alpha",
        type=float,
        default=0.5,
        required=False,
        help="The weighting parameter for secondary structure in score mixing def=0.0",
    )
    ss.add_argument("-t", type=float, default=0.0, help="Threshold of density map1")
    ss.add_argument("-T", type=float, default=0.0, help="Threshold of density map2")
    ss.add_argument("-g", type=float, default=16.0, help="Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]")
    ss.add_argument("-s", type=float, default=7.0, help="Sampling voxel spacing def=7.0")
    ss.add_argument("-A", type=float, default=30.0, help="Sampling angle spacing def=30.0")
    ss.add_argument("-N", type=int, default=10, help="Refine Top [int] models def=10")
    ss.add_argument("-S", action="store_true", default=False, help="Show topN models in PDB format def=false")
    ss.add_argument("-E", type=bool, default=False, help="Evaluation mode of the current position def=false")
    ss.add_argument("-o", type=str, default=None, help="Output folder name")
    ss.add_argument("-nodup", action="store_true", default=False, help="Remove duplicate models using heuristics def=false")
    # prob.add_argument("-B", type=float, default=8.0, help="Bandwidth of the Gaussian filter for probability values def=8.0")
    # prob.add_argument("-R", type=float, default=0.0, help="Threshold for probability values def=0.0")
    ss.add_argument("-gpu", type=int, help="GPU ID to use for CUDA acceleration def=0")
    ss.add_argument("-pdbin", type=str, default=None, help="Input PDB file to be transformed def=None")
    ss.add_argument("-c", type=int, default=2, help="Number of threads to use def=2")
    ss.add_argument("-res", type=float, default=None, help="Resolution of the experimental map used to create simulated map from structure")

    args = parser.parse_args()

    tgt_ss = None

    if args.b.split(".")[-1] == 'pdb' or args.b.split(".")[-1] == 'cif':
        assert args.res is not None, "Please specify resolution when using structure as input."
        # simulate the map at target resolution
        from TEMPy.protein.structure_blurrer import StructureBlurrer
        from TEMPy.protein.structure_parser import PDBParser, mmCIFParser
        sb = StructureBlurrer()
        if args.b.split(".")[-1] == "pdb":
            structure = PDBParser.read_PDB_file('PDB1', args.b, hetatm=False, water=False)
        elif args.b.split(".")[-1] == "cif":
            structure = mmCIFParser.read_mmCIF_file(args.b)
        else:
            raise Exception("Only PDB and mmCIF files are supported for structure input.")
        sim_map = sb.gaussian_blur_real_space(prot=structure, resolution=args.res)
        os.makedirs("tmp_data", exist_ok=True)
        sim_map.write_to_MRC_file("tmp_data/simu_map.mrc")
        assert os.path.exists("tmp_data/simu_map.mrc"), "Failed to create simulated map from structure."
        # set args.b to the simu map
        if args.command == "ss":
            if args.res is None:
                raise ValueError("Please specify resolution when using structure as input.")
            from ssutils.pdb2ss import gen_npy
            print("Generating secondary structure assignment for input structure...")
            tgt_ss = gen_npy(args.b, args.res, verbose=True)
        args.b = "tmp_data/simu_map.mrc"

    assert os.path.exists(args.a), "Reference map not found, please check -a option"
    assert os.path.exists(args.b), "Target map not found, please check -b option"

    # GPU settings
    device = None
    if args.gpu is not None:
        use_gpu = True
        gpu_id = args.gpu
        import torch

        torch.set_grad_enabled(False)
        if not torch.cuda.is_available():
            print("GPU is specified but CUDA is not available.")
            exit(1)
        else:
            # set up torch cuda device
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id} for CUDA acceleration. GPU Name: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("Using FFTW3 for CPU.")
        use_gpu = False

    if args.command == "orig":

        modeVal = Mode[args.M].value

        if args.S and modeVal != "VecProduct":
            print("Warning: -S option is only available in VecProduct mode, ignoring -S option")
            args.S = False

        # print the command used
        print("### Command ###")
        print(" ".join(sys.argv))

        print("### Input Params Summary ###")
        print("Reference Map Path: ", args.a)
        print("Target Map Path: ", args.b)
        print("Threshold of Reference Map: ", args.t)
        print("Threshold of Target Map: ", args.T)
        print("Bandwidth of the Gaussian filter: ", args.g)
        print("Search space voxel spacing: ", args.s)
        print("Search space angle spacing: ", args.A)
        print("Refine Top ", args.N, " models")
        print("Show topN models in PDB format: ", args.S)
        print("Remove duplicates: ", args.nodup)
        print("Scoring mode: ", modeVal)
        print("Number of threads to use: ", args.c)
        if args.o:
            print("Output folder: ", args.o)
        if args.gpu:
            print("Using GPU ID: ", args.gpu)
        if not args.pdbin or not os.path.exists(args.pdbin):
            print("No input PDB file, skipping transformation")
            args.pdbin = None
        else:
            print("Transform PDB file: ", args.pdbin)
        if args.ca and args.ldp and os.path.exists(args.ca) and os.path.exists(args.ldp):
            print("LDP Recall Reranking Enabled")
            print("LDP PDB file: ", args.ldp)
            print("Backbone PDB file: ", args.ca)
        else:
            print("LDP Recall Reranking Disabled")
        if args.al is not None:
            print("Angle limit for searching: ", args.al)

        if args.ldp or args.ca:
            assert args.ldp and args.ca, "Please specify both -ldp and -ca options"
        if args.ldp:
            assert os.path.exists(args.ldp), "LDP file not found, please check -ldp option"
        if args.ca:
            assert os.path.exists(args.ca), "CA file not found, please check -ca option"

        init_start = time.time()

        # construct mrc objects
        ref_map = EMmap(args.a)
        tgt_map = EMmap(args.b)

        # set voxel size
        ref_map.set_vox_size(thr=args.t, voxel_size=args.s)
        tgt_map.set_vox_size(thr=args.T, voxel_size=args.s)

        # unify dimensions
        unify_dims([ref_map, tgt_map], voxel_size=args.s)

        init_end = time.time()
        print("Initialization time: ", init_end - init_start, "s")

        # resample the maps using mean-shift with Gaussian kernel and calculate the vector representation
        print()
        print("###Processing Reference Map Resampling###")
        ref_map.resample_and_vec(dreso=args.g)
        print("###Processing Target Map Resampling###")
        tgt_map.resample_and_vec(dreso=args.g)
        print()

        resample_end = time.time()
        print("Resampling time: ", resample_end - init_end, "s")

        # set mrc output path
        trans_mrc_path = args.b if args.mrcout else None

        print()
        fitter = MapFitter(
            ref_map,
            tgt_map,
            args.A,
            modeVal,
            args.nodup,
            args.ldp,
            args.ca,
            args.pdbin,
            args.c,
            use_gpu,
            device,
            topn=args.N,
            outdir=args.o,
            save_mrc=args.mrcout,
            alpha=None,
            confine_angles=args.al,
            save_vec=args.S,
        )

        fit_init_end = time.time()
        print("Fit initialization time: ", fit_init_end - resample_end, "s")
        fitter.fit()
        fit_end = time.time()
        print("Fit time: ", fit_end - fit_init_end, "s")

    elif args.command == "ss":

        from ssutils.pdb2ss import gen_npy

        print("### Input Params Summary ###")
        print("Reference Map Path: ", args.a)
        print("Target Map Path: ", args.b)
        # print("Target Secondary Structure Assignment: ", args.npa)
        # print("Search Secondary Structure Assignment: ", args.npb)
        print("Threshold of Reference Map: ", args.t)
        print("Threshold of Target Map: ", args.T)
        print("Bandwidth of the Gaussian filter: ", args.g)
        print("Search space voxel spacing: ", args.s)
        print("Search space angle spacing: ", args.A)
        print("Refine Top ", args.N, " models")
        print("Show topN models in PDB format: ", args.S)
        print("Remove duplicates: ", args.nodup)
        print("Number of threads to use: ", args.c)
        if args.o:
            print("Output folder: ", args.o)
        if args.gpu:
            print("Using GPU ID: ", args.gpu)
        if not args.pdbin or not os.path.exists(args.pdbin):
            print("No input PDB file, skipping transformation")
            args.pdbin = None
        else:
            print("Transform PDB file: ", args.pdbin)

        # construct mrc objects
        ref_map = EMmap(args.a, ss_data=np.load(args.npa).astype(np.float32))
        # generate numpy array for target map if input is a PDB file
        if tgt_ss is not None:
            tgt_map = EMmap(args.b, ss_data=tgt_ss.astype(np.float32))
        else:
            tgt_map = EMmap(args.b, ss_data=np.load(args.npb).astype(np.float32))

        # set voxel size
        ref_map.set_vox_size(thr=args.t, voxel_size=args.s)
        tgt_map.set_vox_size(thr=args.T, voxel_size=args.s)

        # unify dimensions
        unify_dims([ref_map, tgt_map], voxel_size=args.s)

        print("\n###Processing Reference Map Resampling###")
        ref_map.resample_and_vec(dreso=args.g)
        print("\n###Processing Target Map Resampling###")
        tgt_map.resample_and_vec(dreso=args.g)
        print()

        fitter = MapFitter(
            ref_map,
            tgt_map,
            args.A,
            "VecProduct",
            args.nodup,
            None,
            None,
            args.pdbin,
            args.c,
            use_gpu,
            device,
            topn=args.N,
            outdir=args.o,
            save_mrc=False,
            alpha=args.alpha,
            save_vec=args.S,
        )

        fitter.fit_ss()
