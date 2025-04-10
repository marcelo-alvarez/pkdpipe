import numpy as np
import matplotlib.pyplot as plt
from importlib.resources import files

from pkdpipe.cosmology import Cosmology

cosmo  = Cosmology(cosmology='euclid-flagship')
cosmo0 = Cosmology(cosmology='euclid-flagship',mnu=0)

h  = cosmo.params['h']
As = cosmo.params['As']
ns = cosmo.params['ns']

k0 = 0.05 / h # pivot scale in h/Mpc

# get pkdgrav3 example file
tfile = files("pkdpipe.data").joinpath("euclid_z0_transfer_combined.dat")
ek, te = np.loadtxt(tfile,unpack=True)

# camb p(k) 
k   = cosmo.k            # h/Mpc
p   = cosmo.pk  / h**3   # Mpc^3
t   = cosmo.T   / h**1.5 # Mpc^(3/2)
p0  = cosmo0.pk / h**3   # Mpc^3
t0  = cosmo0.T  / h**1.5 # Mpc^(3/2)

# pkdgrav3 example file p(k) from T(k)
pe=te**2*As*(ek/k0)**ns # Mpc^3

pe=np.exp(np.interp(np.log(k),np.log(ek),np.log(pe)))
te=np.exp(np.interp(np.log(k),np.log(ek),np.log(te)))

s=13
plt.semilogx(k,(t /te-1),c='k',label=r'$\Sigma{m}_\nu=0.058\ {\rm eV}$')
plt.semilogx(k,(t0/te-1),c='r',label=r'$\Sigma{m}_\nu=0$')
plt.semilogx(k,k*0,c='k',ls=':',lw=0.5)
plt.legend()
plt.gca().set_xlabel(r'$k\ [h/{\rm Mpc}]$',size=s)
plt.gca().set_ylabel(r'$T_{\rm camb}/T_{\rm pkdgrav}-1$',size=s)
plt.gca().set_ylim((-0.009,0.005))
plt.savefig('camb-test.png',dpi=300,bbox_inches='tight')

cosmo.writetransfer('camb-transfer.txt')