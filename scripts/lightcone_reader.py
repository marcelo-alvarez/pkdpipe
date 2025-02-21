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

def read_step(n_step,namespace,zoutput,zout,bounds):
    parts = np.zeros((6,0),dtype=np.float32)
    current_proc = 0
    while True:
        # if we expect everything to be empty just remove the files
        if zoutput[n_step] > zout:
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
                z_0 = zoutput[n_step]
                if np.abs(z_0) <= 1e-9:
                    z_0 = 0.0
                z_1 = zoutput[n_step - 1]
                if z_1 > zout:
                    z_1 = zout

                # break the loop
                print(f"completed step {n_step:>4}",end='\r')
                break

    return parts

def parse_param_file(content,varname,vartype=str):
    if vartype == str:
        return re.search(r'(?<='+varname+r')\s*=\s*["\'].+["\']', content)[0].split("=")[-1].strip().replace('"', '').replace("'", "")
    elif vartype == float:
        return float(re.search(r'(?<='+varname+r')\s*=\s*\d+\.\d+', content)[0].split("=")[-1].strip())
    elif vartype == int:
        return int(re.search(r'(?<='+varname+r')\s*=\s*\d+', content)[0].split("=")[-1].strip())
    else:
        return None

def lightcone_reader(param_file, bounds, nproc):
    # get all the necessary stuff from the param file
    with open(param_file, "r") as f:
        content = f.read()

    namespace = parse_param_file(content,  'achOutName',  str)
    zout      = parse_param_file(content,'dRedshiftLCP',float)
    boxsize   = parse_param_file(content,    'dBoxSize',float)
    steps     = parse_param_file(content,      'nSteps',  int)

    print()
    
    # read out the log file
    zoutput = np.genfromtxt(namespace + ".log")[:, 1]

    args = [[n_step,namespace,zoutput,zout,bounds] for n_step in range(1, steps + 1)]
    with mp.Pool(processes=nproc) as pool:
        multi_collect = pool.starmap(read_step,args)

    allpart = np.zeros((6,0))
    for part in multi_collect:
        if np.shape(part)[1]>0: # is not None:
            allpart = np.concatenate((allpart,part),axis=1)
    print()

    return boxsize,np.rec.fromarrays(allpart.reshape((6,-1)),names=['x','y','z','vx','vy','vz'])