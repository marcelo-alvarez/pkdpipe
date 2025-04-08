import numpy as np
import sys
import os
import camb

"""
pkdpipe cosmology
"""

pkdp_cosmo_defaults = {
    'h':      0.67,
    'omegam': 0.32,
    'omegab': 0.049,
    'omegak': 0.0,
    'sigma8': 0.831,
    'As':     2.1e-9,
    'ns':     0.96,
    'mnu':    0.0587,
    'tau':    0.0543,
    'numnu':  3,
    'harchy': 'normal',
    'k0phys': 0.05, # pivot scale in 1/Mpc
}

class Cosmology:

    '''Cosmology'''
    def __init__(self, **kwargs):
        self.params = {}
        for key in pkdp_cosmo_defaults:
            self.params[key] = kwargs.get(key,pkdp_cosmo_defaults[key])

        h      = self.params['h']
        omegam = self.params['omegam']
        omegab = self.params['omegab']
        omegak = self.params['omegak']
        sigma8 = self.params['sigma8']
        As     = self.params['As']
        ns     = self.params['ns']
        mnu    = self.params['mnu']
        tau    = self.params['tau']
        numnu  = self.params['numnu']
        harchy = self.params['harchy']
        k0phys = self.params['k0phys']
   
        H0    = 100 * h
        ombh2 = omegab * h**2
        omch2 = (omegam - omegab) * h**2
        omk   = omegak
        k0    = k0phys / h # pivot scale in h/Mpc

        self.cambpars = camb.set_params(
            H0=H0,
            As=As,
            ombh2=ombh2,
            omch2=omch2,
            mnu=mnu,
            omk=omk,
            tau=tau,
            ns=ns,
            num_massive_neutrinos=numnu,
            neutrino_hierarchy=harchy)

        self.cambpars.NonLinear = camb.model.NonLinear_none
        self.cambpars.set_matter_power(redshifts=[0,], kmax=10.0)

        self.camb_results = camb.get_results(self.cambpars)
        self.kpk, self.zpk, self.pk = self.camb_results.get_matter_power_spectrum(
            minkh=1e-4, maxkh=10, npoints = 1000)
        self.sigma8_norm = (sigma8 / self.camb_results.get_sigma8_0())**2

        # normalize using sigma8
        self.pk = self.pk[0] * self.sigma8_norm

        # get z=0 transfer function
        self.T = np.sqrt(self.pk/As/(self.kpk/k0)**ns)

