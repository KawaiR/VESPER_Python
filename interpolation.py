import argparse
from distutils import command
from interpolation_functions import *

if __name__ == "__main__":
    print('Usage Modes:\n')
    print('1- Insert: only interpolates the map given.')
    print('2- Graph: interpolates the map given and saves each secondary structure prediction map individually.')
    parser = argparse.ArgumentParser()

    subparser = parser.add_subparsers(dest='command')

    insert = subparser.add_parser('insert')
    graph = subparser.add_parser('graph')

    #Interpolation (no graphing menu)
    insert.add_argument('-f', type=str, required=True, help='Map array (.npy)')
    insert.add_argument('-s', type=str, required=True, help='Name for new map to be saved (string)')

    #Graph after interpolating
    graph.add_argument('-f', type=str, required=True, help='Map array (.npy)')
    graph.add_argument('-s', type=str, required=True, help='Name for individual maps (string)')
    graph.add_argument('-m', type=str, required=True, help='Name of density map file. NOTE: VOXEL SPACIING MUST BE 1. (.mrc)')

    args = parser.parse_args()

    if args.command == 'insert':
        data_file = args.f
        to_save = args.s

        orig_data = np.load(data_file)
        np.set_printoptions(precision=3)

        print(orig_data.shape[0])
        print(orig_data.shape[1])
        print(orig_data.shape[2])
        print(orig_data.shape[3])

        interpolate(orig_data)

        np.save(to_save, orig_data)
        print('Saved interpolated map as: ' + to_save + '.npy')

    elif args.command == 'graph':
        data_file = args.f
        to_save = args.s
        mrc_name = args.m

        orig_data = np.load(data_file)
        np.set_printoptions(precision=3)

        print(orig_data.shape[0])
        print(orig_data.shape[1])
        print(orig_data.shape[2])
        print(orig_data.shape[3])

        interpolate(orig_data)

        np.save(to_save, orig_data)
        print('Saved interpolated map as: ' + to_save + '.npy')

        ids = [0, 1, 2, 3]
        for i in ids:
            if i == 0:
                identifier = "_0.mrc" 
            elif i == 1:
                identifier = "_1.mrc"
            elif i == 2:
                identifier = "_2.mrc"
            else:
                identifier = "_3.mrc"
            
            conv_prediction_to_mrc(orig_data, mrc_name, i, to_save + identifier)

