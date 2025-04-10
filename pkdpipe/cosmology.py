import numpy as np
import sys
import os
import camb

"""
pkdpipe cosmology
"""

def get_cosmology(cosmoname):
    # default is DESI DR2+Planck+ACT parameters with no nu prior but ignoring mnu by setting mnu=0
    h     = 68.906658 / 100
    ombh2 = 0.022597077
    omch2 = 0.11788858
    As    = 2.1302742e-09
    ns    = 0.97278164
    cosmoparams = {
            'h':      h,
            'omegam': (ombh2+omch2) / h**2,
            'omegab': ombh2 / h**2,
            'As':     As,
            'ns':     ns,
            'TCMB':   2.72548,
            'omegak': 0.0,
            'sigma8': None,
            'mnu':    0,
            'nnu':    3.044,
            'tau':    0.0543,
            'numnu':  3,
            'harchy': 'normal',
            'k0phys': 0.05,
        }
    if cosmoname == 'desi-dr2-planck-act-mnufree':
        pass
    elif cosmoname == 'euclid-flagship':
        cosmoparams['h']      = 0.67
        cosmoparams['omegam'] = 0.32
        cosmoparams['omegab'] = 0.049
        cosmoparams['sigma8'] = 0.831
        cosmoparams['As']     = 2.1e-9
        cosmoparams['ns']     = 0.96
        cosmoparams['mnu']    = 0.0587
    return cosmoparams

class Cosmology:

    '''Cosmology'''
    def __init__(self, **kwargs):

        self.cosmoname = kwargs.get('cosmology','default')
        self.params = get_cosmology(self.cosmoname)
        for key in self.params:
            self.params[key] = kwargs.get(key,self.params[key])
   
        k0   = self.params['k0phys'] / self.params['h'] # pivot scale in h/Mpc
        kmin = 5e-5
        kmax = 2e1

        self.cambpars = camb.set_params(
            TCMB                  = self.params['TCMB'],
            H0                    = 100 * self.params['h'],
            As                    = self.params['As'],
            ombh2                 = self.params['omegab'] * self.params['h']**2,
            omch2                 = (self.params['omegam'] - self.params['omegab']) * self.params['h']**2,
            mnu                   = self.params['mnu'],
            nnu                   = self.params['nnu'],
            omk                   = self.params['omegak'],
            tau                   = self.params['tau'],
            ns                    = self.params['ns'],
            num_massive_neutrinos = self.params['numnu'],
            neutrino_hierarchy    = self.params['harchy'])

        self.cambpars.NonLinear = camb.model.NonLinear_none
        self.cambpars.set_matter_power(redshifts=[0,], kmax=kmax, accurate_massive_neutrino_transfers=True)
        self.camb_results = camb.get_results(self.cambpars)
        self.k, self.z, self.pk = self.camb_results.get_matter_power_spectrum(
            minkh=kmin, maxkh=kmax, npoints = 400)

        self.pk = self.pk[0]

        if self.params['sigma8'] is not None:
            # normalize using sigma8
            self.sigma8_norm = (self.params['sigma8'] / self.camb_results.get_sigma8_0())**2
            self.pk = self.pk * self.sigma8_norm
        else:
            self.params['sigma8'] = self.camb_results.get_sigma8_0()

        # get z=0 transfer function
        self.T = np.sqrt(self.pk/self.params['As']/(self.k/k0)**self.params['ns'])

        # generate summary
        self._generate_cosmosummary()

        # add omegal and sigma8 to cosmo parameters
        self.params['omegal'] = self.camb_results.get_Omega(var='de')

        # redshift and comoving distance grids
        self._zgrid = np.logspace(-5, 6, num=10000)
        self._chigrid = self.camb_results.comoving_radial_distance(self._zgrid)

    def z2chi(self, z):
        return np.interp(z, self._zgrid, self._chigrid)

    def chi2z(self, chi):
        return np.interp(chi, self._chigrid, self._zgrid)

    def _generate_cosmosummary(self):
        self.cosmosummary = "\nValues used in CAMB for this transfer function\n\n"
        width=17
        omegas = {}
        h  = self.cambpars.H0 / 100
        sigma8 = self.camb_results.get_sigma8_0()
        Neff = sum(self.cambpars.nu_mass_degeneracies[:self.cambpars.nu_mass_eigenstates]) + self.cambpars.num_nu_massless
        omega = 0.0
        for var in ['cdm', 'baryon', 'photon', 'neutrino', 'nu', 'de']:
            omegas[var] = self.camb_results.get_Omega(var=var)
            omega += omegas[var]
        self.cosmosummary += f'{"As:":>{width}} {self.cambpars.InitPower.As:1.4e}\n'
        self.cosmosummary += f'{"omega_b h^2:":>{width}} {omegas['baryon']*h**2:1.4e}\n'
        self.cosmosummary += f'{"omega_c h^2:":>{width}} {omegas['cdm']*h**2:1.4e}\n'
        self.cosmosummary += f'{"H0:":>{width}} {self.cambpars.H0:1.4e}\n'
        self.cosmosummary += f'{"ns:":>{width}} {self.cambpars.InitPower.ns:1.4e}\n'
        self.cosmosummary += '\n'
        self.cosmosummary += f'{"sigma8:":>{width}} {sigma8:1.4e}\n'
        self.cosmosummary += f'{"omega_nu:":>{width}} {omegas['nu']:1.4e}\n'
        self.cosmosummary += '\n'
        self.cosmosummary += f'{"Neff:":>{width}} {Neff:1.4e}\n'
        self.cosmosummary += f'{"omega_r,gamma:":>{width}} {omegas['photon']:1.4e}\n'
        self.cosmosummary += f'{"omega_r,nu:":>{width}} {omegas['neutrino']:1.4e}\n'
        self.cosmosummary += '\n'
        self.cosmosummary += f'{"omega_l:":>{width}} {omegas['de']:1.4e}\n'
        self.cosmosummary += f'{"omega_r:":>{width}} {(omegas['neutrino']+omegas['photon']):1.4e}\n'
        self.cosmosummary += f'{"omega_m:":>{width}} {omegas['baryon']+omegas['cdm']+omegas['nu']:1.4e}\n'
        self.cosmosummary += f'{"omega:":>{width}} {omegas['de']+omegas['neutrino']+omegas['photon']+omegas['baryon']+omegas['cdm']+omegas['nu']:1.4e}\n'

    def writetransfer(self,filename):
        header = self.cosmosummary
        header += f'\n'
        header += f' Column 1: k [h/Mpc]\n'
        header += f' Column 2: T(k) = sqrt(P(k,z=0) / As / (k/(0.05 1/Mpc))^ns) [Mpc^(3/2)]\n'
        np.savetxt(filename, np.transpose([self.k, self.T]),fmt='%1.5e', header=header)
