import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path
import ast
from typing import Dict, Any
from pkdpipe.datainterface import DataInterface
from cli import parse_command_line, dict_hash

args = parse_command_line()

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

# get data
pfile = f"__pkdpipe-cache__/plotter-{runcode}.npz"
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