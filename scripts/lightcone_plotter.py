import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path
import ast
from typing import Dict, Any
from pkdpipe.datainterface import DataInterface

def dict_hash(dictionary: Dict[str, Any]) -> str:
    from typing import Dict, Any
    import hashlib
    import json

    """
    MD5 hash of a dictionary
    """
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return str(dhash.hexdigest())[:10]

dparam = "/global/cfs/cdirs/cosmosim/slac/malvarez/pkdgrav3/scaling-tests/runs/N1400-L1050-008gpus/N1400-L1050-008gpus.par"
dbbox  = [[-1,1],[-1,1],[-0.0005,0.0005]]
dnproc=12
dcname = None
dpsize = 0.02
ddpi   = 900
dcmap  = "seismic"
ddtype = "particles"

# parse command line
parser = argparse.ArgumentParser(description='Collects the lightcone output of pkdgrav to shells.')
parser.add_argument('--param', type=str,              help=f'Path to the param file used to start the sim', default=dparam)
parser.add_argument('--bbox',  type=ast.literal_eval, help=f'bbox [[xmn,xmx],[ymn,ymx],[zmn,zmx]]; default: {dbbox}', default=dbbox)
parser.add_argument('--nproc', type=int,              help=f'nproc: {dnproc}', default=dnproc)
parser.add_argument('--cname', type=str,              help=f'cname: output in cname.png', default=dcname)
parser.add_argument('--psize', type=float,            help=f'particle size in plot', default=dpsize)
parser.add_argument('--dpi',   type=int,              help=f'dpi of output image', default=ddpi)
parser.add_argument('--cmap',  type=str,              help=f'colormap of output image', default=dcmap)
parser.add_argument('--dtype', type=str,              help=f'data type', default=ddtype)

args = parser.parse_args()
if args.cname is None:
    args.cname = args.dtype
    
pkdata = DataInterface(param_file=args.param,nproc=args.nproc)

boxsize = pkdata.params['boxsize']
runcode=f"{dict_hash(vars(args))}"
outfile = f"{args.cname}.png"

if args.dtype == "particles":
    dataset='xvp'
    filetype='lcp'
elif args.dtype == "halos":
    dataset='xvh'
    filetype='fof'

# get particles
pfile = f"__pkdpipe-cache__/{runcode}.npz"
os.makedirs(os.path.dirname(pfile), exist_ok=True)
if os.path.exists(pfile):
    print(f"using cached data in {pfile}")
    data = np.load(pfile)
    parts = data['parts']
    boxsize = data['boxsize'][0]
else:
    parts = pkdata.fetch_data(args.bbox,dataset=dataset,filetype=filetype)
    print(f"writing cached data to {pfile}")
    np.savez(pfile,parts=parts,boxsize=np.asarray([boxsize]))

boxsize /= 1e3 # Mpc/h --> Gpc/h

x = boxsize * parts['x']
y = boxsize * parts['y']
z = boxsize * parts['z']
vx = parts['vx']
vy = parts['vy']

npart = len(x)

vr = (x*vx + y*vy)/np.sqrt(x**2+y**2)
vr /= (0.3*vr.max())

plt.gca().set_facecolor('black')
plt.scatter(x,y,facecolors=vr,s=args.psize,linewidth=0,c=vr,edgecolors=None,cmap=args.cmap,vmin=-1,vmax=1)
plt.gca().set_aspect('equal')

plt.gca().set_xlim((boxsize*args.bbox[0][0],boxsize*args.bbox[0][1]))
plt.gca().set_ylim((boxsize*args.bbox[1][0],boxsize*args.bbox[1][1]))

plt.gca().set_xlabel('x [Gpc/h]')
plt.gca().set_ylabel('y [Gpc/h]')

plt.savefig(outfile,dpi=args.dpi)