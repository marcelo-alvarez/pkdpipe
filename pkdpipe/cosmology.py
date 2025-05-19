import numpy as np
import camb
from typing import Dict, Any, Union

from .config import COSMOLOGY_PRESETS, DEFAULT_COSMOLOGY_NAME, PkdpipeConfigError

"""
pkdpipe cosmology module.

This module defines the Cosmology class, which interfaces with CAMB
to calculate cosmological parameters, power spectra, and transfer functions.
It uses presets defined in config.py and allows for overriding parameters.
"""

class CosmologyError(PkdpipeConfigError):
    """Custom exception for errors within the Cosmology class."""
    pass


class Cosmology:
    """
    Handles cosmological calculations using CAMB.

    Attributes:
        params (Dict[str, Any]): Dictionary of cosmological parameters used.
        cambpars (camb.CAMBparams): CAMB parameters object.
        camb_results (camb.results.CAMBdata): Results from CAMB calculation.
        k (np.ndarray): Array of k values (wavenumbers).
        pk (np.ndarray): Array of P(k) values (power spectrum at z=0).
        T (np.ndarray): Transfer function T(k) at z=0.
        cosmosummary (str): A string summarizing the cosmology used by CAMB.
    """

    def __init__(self, cosmology: str = DEFAULT_COSMOLOGY_NAME, **kwargs: Any) -> None:
        """
        Initializes the Cosmology object, sets up CAMB, and computes parameters.

        Args:
            cosmology: The name of the cosmology preset to use (from config.COSMOLOGY_PRESETS).
                       Defaults to DEFAULT_COSMOLOGY_NAME.
            **kwargs: Additional cosmological parameters to override the preset values.
                      These should match keys in COSMOLOGY_PRESETS or CAMB parameters.
                      Examples: h=0.7, omegam=0.25, As=2.2e-9, etc.

        Raises:
            CosmologyError: If the specified cosmology preset is not found or if CAMB fails.
        """
        self.cosmoname = cosmology
        self.params: Dict[str, Any] = self._load_cosmology_params(cosmology, **kwargs)

        # CAMB calculation constants
        self._k_pivot_h_mpc: float = self.params['k0phys'] / self.params['h']  # Pivot scale in h/Mpc
        self._kmin_h_mpc: float = 5e-5  # Minimum k for power spectrum
        self._kmax_h_mpc: float = 2e1   # Maximum k for power spectrum
        self._pk_npoints: int = 400     # Number of points for P(k)

        try:
            self._setup_camb_parameters()
            self._run_camb()
            self._process_camb_results()
            self._generate_cosmosummary() # Generate summary after results are processed
            self._setup_interpolation_grids()
        except camb.CAMBError as e:
            raise CosmologyError(f"CAMB calculation failed for cosmology '{self.cosmoname}': {e}")
        except Exception as e:
            raise CosmologyError(f"Unexpected error during Cosmology initialization for '{self.cosmoname}': {e}")

    def _load_cosmology_params(self, preset_name: str, **overrides: Any) -> Dict[str, Any]:
        """
        Loads cosmology parameters from a preset and applies overrides.
        Also ensures derived parameters like ombh2 and omch2 are consistent.
        """
        if preset_name not in COSMOLOGY_PRESETS:
            print(f"Warning: Cosmology preset '{preset_name}' not found. Using default '{DEFAULT_COSMOLOGY_NAME}'.")
            preset_name = DEFAULT_COSMOLOGY_NAME
            if preset_name not in COSMOLOGY_PRESETS:
                 raise CosmologyError(f"Default cosmology preset '{DEFAULT_COSMOLOGY_NAME}' also not found in COSMOLOGY_PRESETS.")

        params = COSMOLOGY_PRESETS[preset_name].copy()

        for key, value in overrides.items():
            if key in params:
                params[key] = value
            else:
                print(f"Info: Overriding/setting parameter '{key}' not in preset '{preset_name}'.")
                params[key] = value
        
        h = params['h']
        if 'omegam' in overrides or 'omegab' in overrides or 'h' in overrides:
            omegab = params.get('omegab', params['ombh2'] / (h**2) if h else 0)
            omegam = params.get('omegam', (params['omch2'] + params['ombh2']) / (h**2) if h else 0)
            
            params['omegab'] = omegab
            params['omegam'] = omegam

            params['ombh2'] = omegab * h**2
            params['omch2'] = (omegam - omegab) * h**2
            
        if params.get('mnu', 0) > 0 and params.get('numnu', 0) == 0:
            print("Warning: mnu > 0 but numnu is 0. Setting numnu to 1 (default for single massive neutrino species).")
            params['numnu'] = 1
        elif params.get('mnu', 0) == 0 and params.get('numnu', 0) != 0:
            pass

        return params

    def _setup_camb_parameters(self) -> None:
        """Sets up CAMB parameters from self.params, compatible with multiple CAMB versions."""
        param_dict = {
            'H0': 100 * self.params['h'],
            'ombh2': self.params['ombh2'],
            'omch2': self.params['omch2'],
            'omk': self.params.get('omegak', 0.0),
            'tau': self.params.get('tau'),
            'TCMB': self.params.get('TCMB', 2.7255),
        }
        # Handle neutrino parameters
        param_dict['mnu'] = self.params.get('mnu', 0.0)
        if self.params.get('mnu', 0.0) > 0:
            param_dict['num_massive_neutrinos'] = self.params.get('numnu', 3)
        # Set up CAMBparams
        self.cambpars = camb.CAMBparams()
        self.cambpars.set_cosmology(**param_dict)
        self.cambpars.InitPower.set_params(
            ns=self.params['ns'],
            As=self.params['As']
        )
        self.cambpars.NonLinear = camb.model.NonLinear_none

    def _run_camb(self) -> None:
        """Runs CAMB calculations."""
        self.cambpars.set_matter_power(redshifts=[0.0], kmax=self._kmax_h_mpc, accurate_massive_neutrino_transfers=True)
        self.camb_results = camb.get_results(self.cambpars)

    def _process_camb_results(self) -> None:
        """Processes results from CAMB, calculates P(k) and T(k)."""
        self.k, _, self.pk = self.camb_results.get_matter_power_spectrum(
            minkh=self._kmin_h_mpc, maxkh=self._kmax_h_mpc, npoints=self._pk_npoints
        )
        self.pk = self.pk[0]

        calculated_sigma8 = self.camb_results.get_sigma8_0()
        if self.params.get('sigma8') is not None and self.params['sigma8'] > 0:
            sigma8_target = self.params['sigma8']
            norm_factor = (sigma8_target / calculated_sigma8)**2
            self.pk *= norm_factor
            self.params['sigma8_calculated_by_camb_before_norm'] = calculated_sigma8
            self.params['sigma8_normalization_factor_applied'] = norm_factor
        else:
            self.params['sigma8'] = calculated_sigma8

        self.T = np.sqrt(self.pk / (self.params['As'] * (self.k / self._k_pivot_h_mpc)**self.params['ns']))
        
        self.params['omegal_calculated'] = self.camb_results.get_Omega('de')
        self.params['omegam_calculated'] = self.camb_results.get_Omega('cdm') + self.camb_results.get_Omega('baryon') + self.camb_results.get_Omega('nu')

        self.params['sigma8_final'] = self.camb_results.get_sigma8_0() * np.sqrt(self.params.get('sigma8_normalization_factor_applied',1.0))


    def _setup_interpolation_grids(self, nz_grid: int = 10000, z_max_grid: float = 1100.0) -> None:
        """Sets up redshift and comoving distance grids for interpolation."""
        self._zgrid = np.logspace(-5, np.log10(z_max_grid), num=nz_grid)
        if not hasattr(self, 'camb_results'):
            raise CosmologyError("CAMB results not available for setting up interpolation grids.")
        self._chigrid = self.camb_results.comoving_radial_distance(self._zgrid)

    def z2chi(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Converts redshift(s) to comoving radial distance in Mpc.

        Args:
            z: Redshift or array of redshifts.

        Returns:
            Comoving radial distance(s) in Mpc.
        """
        if not hasattr(self, '_zgrid') or not hasattr(self, '_chigrid'):
            self._setup_interpolation_grids()
        return np.interp(z, self._zgrid, self._chigrid)

    def chi2z(self, chi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Converts comoving radial distance(s) in Mpc to redshift(s).

        Args:
            chi: Comoving radial distance(s) in Mpc.

        Returns:
            Redshift(s).
        """
        if not hasattr(self, '_zgrid') or not hasattr(self, '_chigrid'):
            self._setup_interpolation_grids()
        return np.interp(chi, self._chigrid, self._zgrid)

    def _generate_cosmosummary(self) -> None:
        """
        Generates a string summarizing the cosmological parameters used by CAMB.
        This summary is primarily for the header of the transfer function file.
        """
        if not hasattr(self, 'camb_results') or not hasattr(self, 'cambpars'):
            self.cosmosummary = "Cosmology summary not available (CAMB results pending)."
            return

        summary_lines = ["\nValues used in CAMB for this transfer function (and derived values)\n"]
        width = 25

        h_val = self.cambpars.H0 / 100.0
        summary_lines.append(f"{'H0 [km/s/Mpc]:':<{width}} {self.cambpars.H0:.4f}")
        summary_lines.append(f"{'h (H0/100):':<{width}} {h_val:.4f}")
        summary_lines.append(f"{'Omega_b h^2 (ombh2):':<{width}} {self.cambpars.ombh2:.5f}")
        summary_lines.append(f"{'Omega_c h^2 (omch2):':<{width}} {self.cambpars.omch2:.5f}")
        summary_lines.append(f"{'Omega_k (omk):':<{width}} {self.cambpars.omk:.4f}")
        summary_lines.append(f"{'mnu [eV]:':<{width}} {self.params.get('mnu', 0.0):.4f}")
        # Safely access neutrino parameters for different CAMB versions
        nnu_value = getattr(self.cambpars, 'standard_neutrino_neff', self.params.get('nnu', 3.044))
        summary_lines.append(f"{'N_eff (standard_neff):':<{width}} {nnu_value:.4f}")
        num_massive_nu = getattr(self.cambpars, 'num_massive_neutrinos', self.params.get('numnu', 0))
        summary_lines.append(f"{'Number of massive nu:':<{width}} {num_massive_nu}")
        hierarchy = getattr(self.cambpars, 'neutrino_hierarchy', self.params.get('harchy', 'normal'))
        summary_lines.append(f"{'Neutrino hierarchy:':<{width}} {hierarchy}")
        summary_lines.append(f"{'A_s (scalar amp):':<{width}} {self.cambpars.InitPower.As:.4e}")
        summary_lines.append(f"{'n_s (scalar index):':<{width}} {self.cambpars.InitPower.ns:.4f}")
        summary_lines.append(f"{'k_pivot (1/Mpc):':<{width}} {self.cambpars.InitPower.pivot_scalar:.4f}")
        summary_lines.append(f"{'k_pivot_h (h/Mpc):':<{width}} {self._k_pivot_h_mpc:.4f}")

        summary_lines.append("\nDerived cosmological densities from CAMB (at z=0):")
        omega_b_0 = self.camb_results.get_Omega('baryon')
        omega_c_0 = self.camb_results.get_Omega('cdm')
        omega_m_0 = self.camb_results.get_Omega('cdm') + self.camb_results.get_Omega('baryon') + self.camb_results.get_Omega('nu')
        omega_l_0 = self.camb_results.get_Omega('de')
        omega_r_0 = self.camb_results.get_Omega('photon') + self.camb_results.get_Omega('neutrino')
        omega_nu_0_massive = self.camb_results.get_Omega('nu')
        
        summary_lines.append(f"{'Omega_baryon (z=0):':<{width}} {omega_b_0:.5f}")
        summary_lines.append(f"{'Omega_cdm (z=0):':<{width}} {omega_c_0:.5f}")
        summary_lines.append(f"{'Omega_nu_massive (z=0):':<{width}} {omega_nu_0_massive:.5e}")
        summary_lines.append(f"{'Omega_matter_total (z=0):':<{width}} {omega_m_0:.5f}")
        summary_lines.append(f"{'Omega_radiation_total (z=0):':<{width}} {omega_r_0:.5e}")
        summary_lines.append(f"{'Omega_DE (Lambda, z=0):':<{width}} {omega_l_0:.5f}")
        
        omega_total = omega_m_0 + omega_l_0 + omega_r_0 + self.cambpars.omk
        summary_lines.append(f"{'Omega_total (sum + omk):':<{width}} {omega_total:.5f}")

        summary_lines.append("\nOther derived values from CAMB:")
        summary_lines.append(f"{'sigma_8 (CAMB calc, z=0):':<{width}} {self.params.get('sigma8_calculated_by_camb_before_norm', self.camb_results.get_sigma8_0()):.4f}")
        if 'sigma8_normalization_factor_applied' in self.params:
            summary_lines.append(f"{'sigma_8 (Target/Input):':<{width}} {self.params.get('sigma8'):.4f}")
            summary_lines.append(f"{'P(k) norm factor (sigma8):':<{width}} {self.params['sigma8_normalization_factor_applied']:.4f}")
        summary_lines.append(f"{'sigma_8 (Final, for P(k)):':<{width}} {self.params.get('sigma8_final', self.camb_results.get_sigma8_0()):.4f}")

        self.cosmosummary = "\n".join(summary_lines)

    def writetransfer(self, filename: str) -> None:
        """
        Writes the calculated transfer function T(k) to a file.
        The file includes a header with the cosmology summary.

        Args:
            filename: The path to the file where the transfer function will be saved.
        
        Raises:
            IOError: If writing to the file fails.
        """
        if not hasattr(self, 'k') or not hasattr(self, 'T'):
            raise CosmologyError("Transfer function T(k) not calculated. Run CAMB first.")

        header = self.cosmosummary
        header += "\n\nColumns in this file:"
        header += f'\n Column 1: k [h/Mpc]'
        header += f'\n Column 2: T(k) = sqrt(P(k,z=0) / (As * (k / k_pivot_h)^ns)) [unitless, if P(k) is (Mpc/h)^3 and As is (Mpc/h)^3]'
        header += f'\n Units of T(k) are (Mpc/h)^(3/2) if As is dimensionless and P(k) is in (Mpc/h)^3.'

        try:
            np.savetxt(filename, np.transpose([self.k, self.T]), fmt='%1.8e', header=header)
            print(f"Transfer function written to '{filename}'.")
        except IOError as e:
            raise IOError(f"Failed to write transfer function to '{filename}': {e}")

# Example usage (for testing)
if __name__ == '__main__':
    print("Testing Cosmology class...")
    try:
        print("\n--- Testing Default Cosmology ---")
        cosmo_default = Cosmology()
        print(cosmo_default.cosmosummary)
        cosmo_default.writetransfer("default_transfer_function.dat")
        print(f"Default: z=1, chi={cosmo_default.z2chi(1.0):.2f} Mpc")
        print(f"Default: chi=3000 Mpc, z={cosmo_default.chi2z(3000.0):.2f}")

        print("\n--- Testing Euclid Flagship Preset ---")
        cosmo_euclid = Cosmology(cosmology='euclid-flagship')
        print(cosmo_euclid.cosmosummary)
        cosmo_euclid.writetransfer("euclid_flagship_transfer_function.dat")

        print("\n--- Testing Custom Overrides (based on default) ---")
        cosmo_custom = Cosmology(
            cosmology=DEFAULT_COSMOLOGY_NAME,
            h=0.7, 
            omegam=0.25, 
            sigma8=0.85,
            ns=0.95
        )
        print(cosmo_custom.cosmosummary)
        cosmo_custom.writetransfer("custom_transfer_function.dat")
        
        print("\n--- Testing Non-existent preset (should use default with warning) ---")
        cosmo_nonexist = Cosmology(cosmology='nonexistent-preset', h=0.72)
        print(cosmo_nonexist.cosmosummary)
        cosmo_nonexist.writetransfer("nonexistent_preset_transfer_function.dat")

        print("\nAll Cosmology tests completed.")

    except PkdpipeConfigError as e:
        print(f"Pkdpipe Configuration Error: {e}")
    except CosmologyError as e:
        print(f"Cosmology Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")
