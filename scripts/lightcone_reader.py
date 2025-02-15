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

def process_single(n_step,namespace,z_output,lcp_type,z_out,bounds):
    cparts = []
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
                parts = np.fromfile(current_file, count=-1, dtype=lcp_type)
                parts = recfc.drop_fields(parts, ["mass","eps","phi"], usemask=False)                    
                # add group parts to the other (from tools in pkdgrav3)
                #assert np.sum(data[0::3]) == 0, "There are apparently grouped parts..."
                cparts.append(parts)
                current_proc += 1
            else:
                # concat and truncate
                cparts = np.concatenate(cparts)

                # save the file with the correct naming scheme
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
    if len(cparts) == 0:
        return None
    lcp_type = np.dtype([("x",'f'),("y",'f'),("z",'f'),("vx",'f'),("vy",'f'),("vz",'f')])
    cparts = cparts.astype(lcp_type)
    dm = ((cparts['x']>bounds[0][0]) & (cparts['x']<bounds[0][1]) &
          (cparts['y']>bounds[1][0]) & (cparts['y']<bounds[1][1]) &
          (cparts['z']>bounds[2][0]) & (cparts['z']<bounds[2][1]))
    cparts = np.vstack((cparts['x'][dm],   cparts['y'][dm],  cparts['z'][dm],
            cparts['vx'][dm], cparts['vy'][dm], cparts['vz'][dm]))
    return cparts


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

    nside = int(re.search(r"(?<=nSideHealpix)\s*=\s*\d+", content)[0].split("=")[-1].strip())    
    print(f"nSideHealpix: ", nside, flush=True)
    z_out = float(re.search(r"(?<=dRedshiftLCP)\s*=\s*\d+\.\d+", content)[0].split("=")[-1].strip())
    print(f"dRedshiftLCP: ", z_out, flush=True)
    steps = int(re.search(r"(?<=nSteps)\s*=\s*\d+", content)[0].split("=")[-1].strip())
    print(f"nSteps:       ", steps, flush=True)
    print()
    
    # read out the log file
    z_output = np.genfromtxt(namespace + ".log")[:, 1]
    lcp_type = np.dtype([("mass",'d'),( "x",'f'),( "y",'f'),( "z",'f'),
                                     ("vx",'f'),("vy",'f'),("vz",'f'),
                                     ("eps",'f'),("phi",'f')])

    args = [[n_step,namespace,z_output,lcp_type,z_out,bounds] for n_step in range(1, steps + 1)]
    with mp.Pool(processes=nproc) as pool:
        multi_collect = pool.starmap(process_single,args)

    allpart = np.zeros((6,0))
    for p in multi_collect:
        part = p#.get()
        if part is not None:
            allpart = np.concatenate((allpart,part),axis=1)
    print()
    return np.rec.fromarrays(allpart,names=['x','y','z','vx','vy','vz'])
