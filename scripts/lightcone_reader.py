import multiprocessing as mp
import numpy as np
import argparse
import sys
import os
import re
import numpy.lib.recfunctions as recfc

"""
Collects the lightcone output and renames the files
"""

def process_single(n_step,namespace,z_output,z_out,bounds):
    parts = np.zeros((6,0),dtype=np.float32)
    current_proc = 0
    while True:
        # if we expect everything to be empty just remove the files
        if z_output[n_step] > z_out:
            break

        # collect and rename
        else:
            # get the current file
            current_file = os.path.join(namespace + ".%05d.lcp.%i" % (n_step, current_proc))
            if os.path.exists(current_file):
                """
                The output in the lcp file contains particle positions and velocities
                """
                cparts = np.fromfile(current_file, dtype=np.float32).reshape((-1,10))
                dm = ((cparts[:,2]>bounds[0][0]) & (cparts[:,2]<bounds[0][1]) &
                      (cparts[:,3]>bounds[1][0]) & (cparts[:,3]<bounds[1][1]) &
                      (cparts[:,4]>bounds[2][0]) & (cparts[:,4]<bounds[2][1]))
                cparts = cparts[dm,2:8].transpose()
                if np.shape(cparts)[1] > 0:
                    parts = np.concatenate((parts,cparts),axis=1)
                current_proc += 1
            elif current_proc > 1:
                # update shell boundaries
                z_0 = z_output[n_step]

                # catch very small z
                if np.abs(z_0) <= 1e-9:
                    z_0 = 0.0

                z_1 = z_output[n_step - 1]

                # catch larger than output redshifts
                if z_1 > z_out:
                    z_1 = z_out

                # break the loop
                print(f"completed step {n_step:>4}",end='\r')
                break

    return parts


def lightcone_reader(param_file, bounds, nproc):
    # get all the necessary stuff from the param file
    with open(param_file, "r") as f:
        content = f.read()

    # some regex stuff
    print(f"Extracting params from: ", param_file, flush=True)
    namespace = re.search(r'(?<=achOutName)\s*=\s*["\'].+["\']', content)[0].split("=")[-1].strip().replace('"', '').replace("'", "")
    print(f"achOutName:   ", namespace, flush=True)

    lcfile = f"{namespace}.lcp"
    if os.path.exists(lcfile):
        return lcfile

    z_out = float(re.search(r"(?<=dRedshiftLCP)\s*=\s*\d+\.\d+", content)[0].split("=")[-1].strip())
    print(f"dRedshiftLCP: ", z_out, flush=True)
    steps = int(re.search(r"(?<=nSteps)\s*=\s*\d+", content)[0].split("=")[-1].strip())
    print(f"nSteps:       ", steps, flush=True)
    boxsize = int(re.search(r"(?<=dBoxSize)\s*=\s*\d+", content)[0].split("=")[-1].strip())
    print(f"BoxSize:      ", boxsize, flush=True)
    print()
    
    # read out the log file
    z_output = np.genfromtxt(namespace + ".log")[:, 1]

    args = [[n_step,namespace,z_output,z_out,bounds] for n_step in range(1, steps + 1)]
    with mp.Pool(processes=nproc) as pool:
        multi_collect = pool.starmap(process_single,args)

    allpart = np.zeros((6,0))
    for part in multi_collect:
        if np.shape(part)[1]>0: # is not None:
            allpart = np.concatenate((allpart,part),axis=1)
    print()

    return boxsize,np.rec.fromarrays(allpart.reshape((6,-1)),names=['x','y','z','vx','vy','vz'])