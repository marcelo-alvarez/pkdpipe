import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os.path
from pkdpipe.datainterface import DataInterface
import ast
import sys

dbounds = [[0,1],[0,1],[-0.0005,0.0005]]
dnproc=12
dcname = "test"
dpsize = 0.02
ddpi   = 900

# parse command line
parser = argparse.ArgumentParser(description='Collects the lightcone output of pkdgrav to shells.')
parser.add_argument('--param', type=str,              help='Path to the param file used to start the sim', required=True)
parser.add_argument('--bounds',type=ast.literal_eval, help=f'bounds [[xmn,xmx],[ymn,ymx],[zmn,zmx]]; default: {dbounds}', default=dbounds)
parser.add_argument('--nproc', type=int,              help=f'nproc: {dnproc}', default=dnproc)
parser.add_argument('--cname', type=str,              help=f'cname: particles cached in cname.npy output in cname.png', default=dcname)
parser.add_argument('--psize', type=float,            help=f'particle size in plot', default=dpsize)
parser.add_argument('--dpi',   type=int,              help=f'dpi of output image', default=ddpi)

args = parser.parse_args()
pkdata = DataInterface(param_file=args.param,nproc=args.nproc)

boxsize = pkdata.params['boxsize']

# get particles
pfile = f"{args.cname}.npz"
if os.path.exists(pfile):
    print(f"using cached particles in {pfile}")
    data = np.load(pfile)
    parts = data['parts']
    boxsize = data['boxsize'][0]
else:
    parts = pkdata.fetch_data(args.bounds,dataset='xv',filetype='lcp')
    np.savez(pfile,parts=parts,boxsize=np.asarray([boxsize]))

boxsize /= 1e3 # Mpc/h --> Gpc/h

x = boxsize * parts['x']
y = boxsize * parts['y']
z = boxsize * parts['z']
vx = parts['vx']
vy = parts['vy']

npart = len(x)

vr = (x*vx + y*vy)/np.sqrt(x**2+y**2)
vr /= vr.max()

cmap='seismic'
plt.gca().set_facecolor('black')
plt.scatter(x,y,facecolors=vr,s=args.psize,linewidth=0,c=vr,edgecolors=None,cmap=cmap)
plt.gca().set_aspect('equal')

plt.gca().set_xlim((0,boxsize))
plt.gca().set_ylim((0,boxsize))

plt.gca().set_xlabel('x [Gpc/h]')
plt.gca().set_ylabel('y [Gpc/h]')

plt.savefig(args.cname,dpi=args.dpi)