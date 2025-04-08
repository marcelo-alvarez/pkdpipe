import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files

from pkdpipe.datainterface import DataInterface
from pkdpipe.cosmology import Cosmology

cosmo = Cosmology()
cosmo0 = Cosmology(mnu=0)

h = cosmo.params['h']
ns = cosmo.params['ns']
As = cosmo.params['As']
k0 = 0.05 / h # pivot scale in h/Mpc

# get
tfile = files("pkdpipe.data").joinpath("euclid_z0_transfer_combined.dat")
ek, te = np.loadtxt(tfile,unpack=True)

# camb p(k) 
k  = cosmo.kpk # h/Mpc
p = cosmo.pk / h**3 # Mpc^3
t  = cosmo.T / h**1.5 # Mpc^(3/2)
p0 = cosmo0.pk / h**3 # Mpc^3
t0  = cosmo0.T / h**1.5 # Mpc^(3/2)

# pkdgrav3 example file p(k) from T(k)
pe=te**2*As*(ek/k0)**ns # Mpc^3

pe=np.exp(np.interp(np.log(k),np.log(ek),np.log(pe)))
te=np.exp(np.interp(np.log(k),np.log(ek),np.log(te)))

s=13
plt.semilogx(k,(p/pe-1),c='k',label=r'$\Sigma{m}_\nu=0.058\ {\rm eV}$')
plt.semilogx(k,(p0/pe-1),c='r',label=r'$\Sigma{m}_\nu=0$')
plt.semilogx(k,k*0,c='k',ls=':',lw=0.5)
plt.legend()
plt.gca().set_xlabel(r'$k\ [h/{\rm Mpc}]$',size=s)
plt.gca().set_ylabel(r'$P_{\rm camb}/P_{\rm pkdgrav}-1$',size=s)
plt.savefig('transfer.png',dpi=300,bbox_inches='tight')
