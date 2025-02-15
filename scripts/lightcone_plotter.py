import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os.path
from lightcone_reader import lightcone_reader
import ast
import sys

dbounds = [[-1,1],[-1,1],[-0.002,0.002]]
dnproc=12
dcname = "test"

# parse command line
parser = argparse.ArgumentParser(description='Collects the lightcone output of pkdgrav to shells.')
parser.add_argument('--param', type=str,              help='Path to the param file used to start the sim', required=True)
parser.add_argument('--bounds',type=ast.literal_eval, help=f'bounds [[xmn,xmx],[ymn,ymx],[zmn,zmx]]; default: {dbounds}', default=dbounds)
parser.add_argument('--nproc', type=int,              help=f'nproc: {dnproc}', default=dnproc)
parser.add_argument('--cname', type=str,              help=f'cname: particles cached in cname.npy output in cname.png', default=dcname)

args = parser.parse_args()

# get particles
pfile = f"{args.cname}.npy"
print(pfile)
if os.path.exists(pfile):
    print(f"using cached particles in {pfile}")
    parts = np.load(pfile)
else:
    parts = lightcone_reader(args.param, args.bounds, args.nproc)
    np.save(pfile,parts)

x = parts['x']
y = parts['y']
z = parts['z']
vx = parts['vx']
vy = parts['vy']

vx -= vx.mean() ; vy -= vy.mean()
vr = (x*vx + y*vy)/np.sqrt(x**2+y**2)
vr /= vr.max()
vr[vr<-vr.max()] = vr.max()

basic_cols=['red', 'grey', 'blue']
cmap = LinearSegmentedColormap.from_list(
    "cmap_name",
    basic_cols
)
cmap='seismic'
plt.gca().set_facecolor('black')
plt.scatter(x,y,facecolors=vr,s=0.05,linewidth=0,c=vr,edgecolors=None,cmap=cmap)
plt.gca().set_aspect('equal')

plt.gca().set_xlim((0,1))
plt.gca().set_ylim((0,1))
plt.savefig(args.cname,dpi=900)