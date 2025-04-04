from pkdpipe.datainterface import DataInterface

# number of process
nproc=64

# bounding box to read in
bbox  = [[-1,1],[-1,1],[-0.0005,0.0005]]

# this is the pkdgrav3 parameter file for the run with halos we want to access
simpath = "/global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/scaling-tests/runs/"

param_file = f"{simpath}/N1400-L1050-008gpus/N1400-L1050-008gpus_v0.par"
#param_file = f"{simpath}/N2800-L2100-064gpus/N2800-L2100-064gpus.par"

pkdata = DataInterface(param_file=param_file,nproc=nproc)
parts = pkdata.fetch_data(bbox,dataset='xvh',filetype='fof')

boxsize = pkdata.params['boxsize']

boxsize /= 1e3 # Mpc/h --> Gpc/h

x = boxsize * parts['x']
y = boxsize * parts['y']
z = boxsize * parts['z']
vx = parts['vx']
vy = parts['vy']

