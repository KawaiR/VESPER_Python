import numpy as np
import os
import mrcfile
from numba import jit

@jit(nopython=True)
def interpolate(data):
    found = False
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if (data[i, j, k, 0] > 0):
                    found = True
                    if i % 2 == 0:
                        ref_x = 0
                    else:
                        ref_x = 1

                    if j % 2 == 0:
                        ref_y = 0
                    else:
                        ref_y = 1

                    if k % 2 == 0:
                        ref_z = 0
                    else:
                        ref_z = 1

                    #print(i, j, k)

                if found:
                    break
            if found:
                break
        if found:
            break

    #print(ref_x, ref_y, ref_z)
    #print((ref_x + 1) % 2, (ref_y + 1) % 2, (ref_z + 1) % 2)

    #Change a single into the opposite of the reference at a time
    for i in range((ref_x + 1) % 2, data.shape[0]-1, 2):
        for j in range(ref_y, data.shape[1]-1, 2):
            for k in range(ref_z, data.shape[2]-1, 2):
                for l in range(data.shape[3]):
                    if (data[i, j, k, l] == 0):
                        avg = 0
                        avg += data[i+1, j, k, l]
                        avg += data[i-1, j, k, l]
                        avg = avg / float(2)
                        data[i, j, k, l] = avg


    for j in range((ref_y + 1) % 2, data.shape[1]-1, 2):
        for i in range(ref_x, data.shape[0]-1, 2):
            for k in range(ref_z, data.shape[2]-1, 2):
                for l in range(data.shape[3]):
                    if (data[i, j, k, l] == 0):
                        avg = 0
                        avg += data[i, j+1, k, l]
                        avg += data[i, j-1, k, l]
                        avg = avg / float(2)
                        data[i, j, k, l] = avg


    for k in range((ref_z + 1) % 2, data.shape[2]-1, 2):
        for i in range(ref_x, data.shape[0]-1, 2):
            for j in range(ref_y, data.shape[1]-1, 2):
                for l in range(data.shape[3]):
                    if (data[i, j, k, l] == 0):
                        avg = 0
                        avg += data[i, j, k+1, l]
                        avg += data[i, j, k-1, l]
                        avg = avg / float(2)
                        data[i, j, k, l] = avg


    #Change two indices into the opposite of the reference at a time
    for i in range((ref_x + 1) % 2, data.shape[0]-1, 2):
        for j in range((ref_y + 1) % 2, data.shape[1]-1, 2):
            for k in range(ref_z, data.shape[2]-1, 2):
                for l in range(data.shape[3]):
                    if (data[i, j, k, l] == 0):
                        avg = 0
                        avg += data[i+1, j, k, l]
                        avg += data[i-1, j, k, l]
                        avg += data[i, j+1, k, l]
                        avg += data[i, j-1, k, l]
                        avg = avg / float(4)
                        data[i, j, k, l] = avg


    for i in range((ref_x + 1) % 2, data.shape[0]-1, 2):
        for k in range((ref_z + 1) % 2, data.shape[2]-1, 2):
            for j in range(ref_y, data.shape[1]-1, 2):
                for l in range(data.shape[3]):
                    if (data[i, j, k, l] == 0):
                        avg = 0
                        avg += data[i+1, j, k, l]
                        avg += data[i-1, j, k, l]
                        avg += data[i, j, k+1, l]
                        avg += data[i, j, k-1, l]
                        avg = avg / float(4)
                        data[i, j, k, l] = avg


    for j in range((ref_y + 1) % 2, data.shape[1]-1, 2):
        for k in range((ref_z + 1) % 2, data.shape[2]-1, 2):
            for i in range(ref_x, data.shape[0]-1, 2):
                for l in range(data.shape[3]):
                    if (data[i, j, k, l] == 0):
                        avg = 0
                        avg += data[i, j+1, k, l]
                        avg += data[i, j-1, k, l]
                        avg += data[i, j, k+1, l]
                        avg += data[i, j, k-1, l]
                        avg = avg / float(4)
                        data[i, j, k, l] = avg


    #Change all indices into the opposite of the reference
    for i in range((ref_x + 1) % 2, data.shape[0]-1, 2):
        for j in range((ref_y + 1) % 2, data.shape[1]-1, 2):
            for k in range((ref_z + 1) % 2, data.shape[2]-1, 2):
                for l in range(data.shape[3]):
                    if (data[i, j, k, l] == 0):
                        avg = 0
                        avg += data[i+1, j, k, l]
                        avg += data[i-1, j, k, l]
                        avg += data[i, j+1, k, l]
                        avg += data[i, j-1, k, l]
                        avg += data[i, j, k+1, l]
                        avg += data[i, j, k-1, l]
                        avg = avg / float(6)
                        data[i, j, k, l] = avg

def conv_prediction_to_mrc(arr,reffile,keyidx,outmapfile):
    with mrcfile.open(reffile) as mrc:
        nx,ny,nz = arr.shape[0], arr.shape[1], arr.shape[2]
        mx,my,mz,cella = mrc.header.mx,mrc.header.my,mrc.header.mz, mrc.header.cella


        mrc_new = mrcfile.new(outmapfile,overwrite=True)

        mrc_new.set_data(np.zeros((nz, ny, nx), dtype=np.float32))
        mrc_new.header.nxstart=arr.shape[0]
        mrc_new.header.nystart=arr.shape[1]
        mrc_new.header.nzstart=arr.shape[2]
        mrc_new.header.origin.x = mrc.header.origin.x
        mrc_new.header.origin.y = mrc.header.origin.y
        mrc_new.header.origin.z = mrc.header.origin.z
        mrc_new.header.cella['x'] = cella['x']
        mrc_new.header.cella['y'] = cella['y']
        mrc_new.header.cella['z'] = cella['z']

        print(nx, ny, nz)
        for i in range(nx - 2):
            for j in range(ny - 2):
                for k in range(nz - 2):
                    mrc_new.data[k, j, i] = arr[i, j, k, keyidx]

        vsize=mrc_new.voxel_size
        vsize.flags.writeable = True
        mrc_new.voxel_size=vsize

        mrc_new.update_header_stats()

        print("original", mrc.voxel_size)
        mrc.print_header()

        print()

        print("new", mrc_new.voxel_size)
        mrc_new.print_header()

        print()

        mrc_new.close()