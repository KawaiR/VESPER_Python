import argparse
from enum import Enum

from search import *
from utils import mrc_set_vox_size, resample_and_vec


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
    prob = subparser.add_parser("prob")

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
    orig.add_argument("-I", type=str, default=None, help="Interpolation mode def=None")
    orig.add_argument("-gpu", type=int, help="GPU ID to use for CUDA acceleration def=0")
    orig.add_argument("-nodup", action="store_true", default=False, help="Remove duplicate models using heuristics " "def=false")
    orig.add_argument("-ldp", type=str, default=None, help="Path to the local dense point file def=None")
    orig.add_argument("-ca", type=str, default=None, help="Path to the CA file def=None")
    orig.add_argument("-pdbin", type=str, default=None, help="Input PDB file to be transformed def=None")
    orig.add_argument("-mrcout", action="store_true", default=False, help="Output the transformed query map def=false")
    orig.add_argument("-c", type=int, default=2, help="Number of threads to use def=2")

    # secondary structure matching menu
    prob.add_argument("-a", type=str, required=True, help="MAP1.mrc (large)")
    prob.add_argument("-npa", type=str, required=True, help="numpy array for Predictions for map 1")
    prob.add_argument("-b", type=str, required=True, help="MAP2.mrc (small)")
    prob.add_argument("-npb", type=str, required=True, help="numpy array for Predictions for map 2")
    prob.add_argument(
        "-alpha", type=float, default=0.5, required=False, help="The weighting parameter for alpha " "mixing def=0.0"
    )
    prob.add_argument("-t", type=float, default=0.0, help="Threshold of density map1")
    prob.add_argument("-T", type=float, default=0.0, help="Threshold of density map2")
    prob.add_argument(
        "-g", type=float, default=16.0, help="Bandwidth of the Gaussian filter def=16.0, sigma = 0.5*[value entered]"
    )
    prob.add_argument("-s", type=float, default=7.0, help="Sampling voxel spacing def=7.0")
    prob.add_argument("-A", type=float, default=30.0, help="Sampling angle spacing def=30.0")
    prob.add_argument("-N", type=int, default=10, help="Refine Top [int] models def=10")
    prob.add_argument("-S", action="store_true", default=False, help="Show topN models in PDB format def=false")
    prob.add_argument(
        "-M",
        type=str,
        default="V",
        help="V: vector product mode (default)\n"
        + "O: overlap mode\n"
        + "C: Cross Correlation Coefficient Mode\n"
        + "P: Pearson Correlation Coefficient Mode\n"
        + "L: Laplacian Filtering Mode",
    )
    prob.add_argument("-E", type=bool, default=False, help="Evaluation mode of the current position def=false")
    prob.add_argument("-P", type=int, default=4, help="Number of processors to use def=4")
    prob.add_argument("-vav", type=float, help="Pre-computed average for density map")
    prob.add_argument("-vstd", type=float, help="Pre-computed standard deviation for density map")
    prob.add_argument("-pav", type=float, help="Pre-computed average for probability map")
    prob.add_argument("-pstd", type=float, help="Pre-computed standard deviation for probability map")
    prob.add_argument("-o", type=str, default=None, help="Output folder name")
    prob.add_argument("-I", type=str, default=None, help="Interpolation mode def=None")
    prob.add_argument("-B", type=float, default=8.0, help="Bandwidth of the Gaussian filter for probability values def=8.0")
    prob.add_argument("-R", type=float, default=0.0, help="Threshold for probability values def=0.0")
    prob.add_argument("-gpu", type=int, help="GPU ID to use for CUDA acceleration def=0")

    args = parser.parse_args()

    if not os.path.exists(args.a):
        print("Target map not found, please check -a option")
        exit(-1)
    if not os.path.exists(args.b):
        print("Query map not found, please check -b option")
        exit(-1)

    if args.command == "orig":

        vox_size = args.s
        modeVal = Mode[args.M].value

        print("### Input Params Summary ###")
        print("Reference Map Path: ", args.a)
        print("Target Map Path: ", args.b)
        print("Threshold of Reference Map: ", args.t)
        print("Threshold of Target Map: ", args.T)
        print("Bandwidth of the Gaussian filter: ", args.g)
        print("Sampling voxel spacing: ", vox_size)
        print("Sampling angle spacing: ", args.A)
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

        # construct mrc objects
        ref_map = MrcObj(args.a)
        tgt_map = MrcObj(args.b)

        # set voxel size
        ref_map, resampled_ref_map = mrc_set_vox_size(ref_map, args.t, vox_size)
        tgt_map, resampled_tgt_map = mrc_set_vox_size(tgt_map, args.T, vox_size)

        if resampled_ref_map.xdim > resampled_tgt_map.xdim:
            unified_dim = resampled_tgt_map.xdim = resampled_tgt_map.ydim = resampled_tgt_map.zdim = resampled_ref_map.xdim

            resampled_tgt_map.orig[0] = resampled_tgt_map.cent[0] - 0.5 * vox_size * resampled_tgt_map.xdim
            resampled_tgt_map.orig[1] = resampled_tgt_map.cent[1] - 0.5 * vox_size * resampled_tgt_map.xdim
            resampled_tgt_map.orig[2] = resampled_tgt_map.cent[2] - 0.5 * vox_size * resampled_tgt_map.xdim

        else:
            unified_dim = resampled_ref_map.xdim = resampled_ref_map.ydim = resampled_ref_map.zdim = resampled_tgt_map.xdim

            resampled_ref_map.orig[0] = resampled_ref_map.cent[0] - 0.5 * vox_size * resampled_ref_map.xdim
            resampled_ref_map.orig[1] = resampled_ref_map.cent[1] - 0.5 * vox_size * resampled_ref_map.xdim
            resampled_ref_map.orig[2] = resampled_ref_map.cent[2] - 0.5 * vox_size * resampled_ref_map.xdim

        # init new dens, vec, data arrays
        resampled_ref_map.dens = np.zeros((unified_dim**3, 1), dtype="float32")
        resampled_ref_map.vec = np.zeros((unified_dim, unified_dim, unified_dim, 3), dtype="float32")
        resampled_ref_map.data = np.zeros((unified_dim, unified_dim, unified_dim), dtype="float32")
        resampled_tgt_map.dens = np.zeros((unified_dim**3, 1), dtype="float32")
        resampled_tgt_map.vec = np.zeros((unified_dim, unified_dim, unified_dim, 3), dtype="float32")
        resampled_tgt_map.data = np.zeros((unified_dim, unified_dim, unified_dim), dtype="float32")

        # resample_and_vec
        print("\n###Processing MAP1 Resampling###")
        # mrc_N1 = resample_and_vec(mrc1, mrc_N1, dreso=bandwidth, density_map=mrc1.data)
        resampled_ref_map = resample_and_vec(ref_map, resampled_ref_map, dreso=(args.g))
        print("\n###Processing MAP2 Resampling###")
        # start_time = time.time()
        # mrc_N2 = resample_and_vec(mrc2, mrc_N2, dreso=bandwidth, density_map=mrc2.data)
        resampled_tgt_map = resample_and_vec(tgt_map, resampled_tgt_map, dreso=(args.g))
        # print("--- %s seconds ---" % (time.time() - start_time))
        print()

        device = None
        # set GPU ID
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
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Using GPU {gpu_id} for CUDA acceleration.")
        else:
            print("Using FFTW3 for CPU.")
            use_gpu = False
            gpu_id = -1

        # set mrc output path
        trans_mrc_path = args.b if args.mrcout else None

        search_map_fft(
            resampled_ref_map,
            resampled_tgt_map,
            TopN=args.N,
            ang=args.A,
            mode=modeVal,
            is_eval_mode=args.E,
            showPDB=args.S,
            folder=args.o,
            gpu=use_gpu,
            device=device,
            remove_dup=args.nodup,
            ldp_path=args.ldp,
            backbone_path=args.ca,
            input_pdb=args.pdbin,
            input_mrc=trans_mrc_path,
            threads=args.c,
        )

    elif args.command == "prob":
        # output folder
        folder = args.o

        # mrc file paths
        objA = args.a
        objB = args.b

        # numpy arrays

        npA = args.npa
        npB = args.npb

        # probability map file paths
        probA = args.a
        probB = args.b

        # all optional positional arguments
        # check for none
        threshold1 = args.t
        threshold2 = args.T

        # weighting parameter
        alpha = args.alpha

        # with default
        bandwidth = args.g
        vox_size = args.s
        angle_spacing = args.A
        topN = args.N

        # average and standard deviation
        vave = args.vav
        vstd = args.vstd
        pave = args.pav
        pstd = args.pstd

        if vave is None:
            vave = -10
        if vstd is None:
            vstd = -10
        if pave is None:
            pave = -10
        if pstd is None:
            pstd = -10

        # necessary?
        showPDB = args.S

        modeVal = args.M
        evalMode = args.E

        num_processors = args.P

        print("### Input Params Summary ###")
        print("Target MAP: ", objA)
        print("Search MAP: ", objB)
        print("Target Secondary Structure Assignment: ", args.npa)
        print("Search Secondary Structure Assignment: ", args.npb)
        print("Threshold of Target MAP: ", threshold1)
        print("Threshold of Search MAP: ", threshold2)
        print("Bandwidth of the Gaussian filter: ", bandwidth)
        print("Sampling voxel spacing: ", vox_size)
        print("Sampling angle spacing: ", angle_spacing)
        print("Refine Top ", topN, " models")
        print("Show topN models in PDB format: ", showPDB)
        print("Alpha: ", alpha)
        print("Bandwidth of the Gaussian filter for probability values: ", args.B)
        print("Threshold for probability values: ", args.R)

        prob_maps = np.load(npA).astype(np.float32)
        prob_maps_chain = np.load(npB).astype(np.float32)

        ref_map = MrcObj(objA)
        tgt_map = MrcObj(objB)

        # probability MRCs
        mrc1_p1 = MrcObj(objA)
        mrc1_p2 = MrcObj(objA)
        mrc1_p3 = MrcObj(objA)
        mrc1_p4 = MrcObj(objA)

        mrc2_p1 = MrcObj(objB)
        mrc2_p2 = MrcObj(objB)
        mrc2_p3 = MrcObj(objB)
        mrc2_p4 = MrcObj(objB)

        # fill in the data
        mrc2_p1.data = np.swapaxes(prob_maps_chain[:, :, :, 0], 0, 2)
        mrc2_p2.data = np.swapaxes(prob_maps_chain[:, :, :, 1], 0, 2)
        mrc2_p3.data = np.swapaxes(prob_maps_chain[:, :, :, 2], 0, 2)
        mrc2_p4.data = np.swapaxes(prob_maps_chain[:, :, :, 3], 0, 2)

        mrc1_p1.data = np.swapaxes(prob_maps[:, :, :, 0], 0, 2)
        mrc1_p2.data = np.swapaxes(prob_maps[:, :, :, 1], 0, 2)
        mrc1_p3.data = np.swapaxes(prob_maps[:, :, :, 2], 0, 2)
        mrc1_p4.data = np.swapaxes(prob_maps[:, :, :, 3], 0, 2)

        print("\n###Generating Params for Resampling Density Map 1###")
        ref_map, resampled_ref_map = mrc_set_vox_size(ref_map, threshold1, vox_size)

        print("\n###Generating Params for Resampling Density Map 2###")
        tgt_map, resampled_tgt_map = mrc_set_vox_size(tgt_map, threshold2, vox_size)

        print("\n###Generating Params for Resampling Probability Map 1###")

        mrc1_p1, mrc_N1_p1 = mrc_set_vox_size(mrc1_p1, thr=args.R, voxel_size=vox_size)
        mrc1_p2, mrc_N1_p2 = mrc_set_vox_size(mrc1_p2, thr=args.R, voxel_size=vox_size)
        mrc1_p3, mrc_N1_p3 = mrc_set_vox_size(mrc1_p3, thr=args.R, voxel_size=vox_size)
        mrc1_p4, mrc_N1_p4 = mrc_set_vox_size(mrc1_p4, thr=args.R, voxel_size=vox_size)

        print("\n###Generating Params for Resampling Probability Map 2###")

        mrc2_p1, mrc_N2_p1 = mrc_set_vox_size(mrc2_p1, thr=args.R, voxel_size=vox_size)
        mrc2_p2, mrc_N2_p2 = mrc_set_vox_size(mrc2_p2, thr=args.R, voxel_size=vox_size)
        mrc2_p3, mrc_N2_p3 = mrc_set_vox_size(mrc2_p3, thr=args.R, voxel_size=vox_size)
        mrc2_p4, mrc_N2_p4 = mrc_set_vox_size(mrc2_p4, thr=args.R, voxel_size=vox_size)

        mrc_list = [
            resampled_ref_map,
            resampled_tgt_map,
            mrc_N1_p1,
            mrc_N1_p2,
            mrc_N1_p3,
            mrc_N1_p4,
            mrc_N2_p1,
            mrc_N2_p2,
            mrc_N2_p3,
            mrc_N2_p4,
        ]
        max_dim = np.max([mrc.xdim for mrc in mrc_list])

        # Unify dimensions in all maps

        for mrc in mrc_list:
            if mrc.xdim != max_dim:
                mrc.xdim = mrc.ydim = mrc.zdim = max_dim
                mrc.orig = mrc.cent - 0.5 * vox_size * mrc.xdim

        for mrc in mrc_list:
            mrc.dens = np.zeros((max_dim**3, 1), dtype="float32")
            mrc.data = np.zeros((max_dim, max_dim, max_dim), dtype="float32")
            mrc.vec = np.zeros((max_dim, max_dim, max_dim, 3), dtype="float32")

        # mrc_N1 = resample_and_vec(mrc1, mrc_N1, dreso=bandwidth)
        # mrc_N1_p1 = resample_and_vec(mrc1_p1, mrc_N1_p1, dreso=bandwidth)
        # mrc_N1_p2 = resample_and_vec(mrc1_p2, mrc_N1_p2, dreso=bandwidth)
        # mrc_N1_p3 = resample_and_vec(mrc1_p3, mrc_N1_p3, dreso=bandwidth)
        # mrc_N1_p4 = resample_and_vec(mrc1_p4, mrc_N1_p4, dreso=bandwidth)
        #
        # mrc_N2 = resample_and_vec(mrc2, mrc_N2, dreso=bandwidth)
        # mrc_N2_p1 = resample_and_vec(mrc2_p1, mrc_N2_p1, dreso=bandwidth)
        # mrc_N2_p2 = resample_and_vec(mrc2_p2, mrc_N2_p2, dreso=bandwidth)
        # mrc_N2_p3 = resample_and_vec(mrc2_p3, mrc_N2_p3, dreso=bandwidth)
        # mrc_N2_p4 = resample_and_vec(mrc2_p4, mrc_N2_p4, dreso=bandwidth)

        print("\n###Processing MAP1 Resampling###")

        resampled_ref_map = resample_and_vec(ref_map, resampled_ref_map, dreso=bandwidth)
        mrc_N1_p1 = resample_and_vec(mrc1_p1, mrc_N1_p1, dreso=args.B, density_map=ref_map.data)
        mrc_N1_p2 = resample_and_vec(mrc1_p2, mrc_N1_p2, dreso=args.B, density_map=ref_map.data)
        mrc_N1_p3 = resample_and_vec(mrc1_p3, mrc_N1_p3, dreso=args.B, density_map=ref_map.data)
        mrc_N1_p4 = resample_and_vec(mrc1_p4, mrc_N1_p4, dreso=args.B, density_map=ref_map.data)

        print("\n###Processing MAP2 Resampling###")
        resampled_tgt_map = resample_and_vec(tgt_map, resampled_tgt_map, dreso=bandwidth)
        mrc_N2_p1 = resample_and_vec(mrc2_p1, mrc_N2_p1, dreso=args.B, density_map=tgt_map.data)
        mrc_N2_p2 = resample_and_vec(mrc2_p2, mrc_N2_p2, dreso=args.B, density_map=tgt_map.data)
        mrc_N2_p3 = resample_and_vec(mrc2_p3, mrc_N2_p3, dreso=args.B, density_map=tgt_map.data)
        mrc_N2_p4 = resample_and_vec(mrc2_p4, mrc_N2_p4, dreso=args.B, density_map=tgt_map.data)

        search_map_fft_prob(
            resampled_ref_map,
            resampled_tgt_map,
            mrc_N1_p1,
            mrc_N1_p2,
            mrc_N1_p3,
            mrc_N1_p4,
            mrc_N2_p1,
            mrc_N2_p2,
            mrc_N2_p3,
            mrc_N2_p4,
            ang=angle_spacing,
            alpha=alpha,
            TopN=topN,
            vave=vave,
            vstd=vstd,
            pave=pave,
            pstd=pstd,
            showPDB=showPDB,
            folder=folder,
        )
