from pkdpipe.datainterface import DataInterface

# number of process
nproc=64

# bounding box to read in
bbox  = [[-1,1],[-1,1],[-0.0005,0.0005]]

# this is the pkdgrav3 parameter file for the run with halos we want to access
simpath = "/pscratch/sd/m/malvarez/pkdgrav3/scaling-tests/"

param_file = f"{simpath}/N1400-L1050-008gpus/N1400-L1050-008gpus.par"
#param_file = f"{simpath}/N2800-L2100-064gpus/N2800-L2100-064gpus.par"

pkdata = DataInterface(param_file=param_file,nproc=nproc)
parts = pkdata.fetch_data(bbox,dataset='xvh',filetype='fof')

boxsize  = pkdata.params['boxsize'] / 1e3 # Gpc/h
pmass    = 2.775e20 * pkdata.params['omegam'] * boxsize**3 / pkdata.params['ngrid']**3 # Msun/h

x = boxsize * parts['x']
y = boxsize * parts['y']

vx = parts['vx']
vy = parts['vy']

Mfof = parts['npart'] * pmass # Msun/h