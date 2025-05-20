import numpy as np
import camb
from typing import Dict, Any, Union

from .config import COSMOLOGY_PRESETS, DEFAULT_COSMOLOGY_NAME, PkdpipeConfigError

"""
pkdpipe cosmology module.

This module defines the Cosmology class, which interfaces with CAMB (Code for Anisotropies 
in the Microwave Background) to calculate various cosmological parameters, 
linear matter power spectra, and transfer functions. It is designed to be flexible, 
allowing users to select from predefined cosmology presets (defined in `config.py`) 
or specify custom cosmological parameters.

The `Cosmology` class handles:
- Loading and overriding cosmological parameters.
- Setting up and running CAMB.
- Processing CAMB results to derive P(k), T(k), sigma8, and other relevant quantities.
- Providing utility functions for redshift-to-comoving distance conversions.
- Writing transfer functions to disk in a standardized format.

Core functionalities include:
- Initialization with a named cosmology preset or custom parameters.
- Calculation of P(k) and T(k) at z=0.
- Normalization of P(k) to a target sigma8 if provided.
- Generation of a detailed summary of the cosmology used.
"""

class CosmologyError(PkdpipeConfigError):
    """Custom exception for errors specific to the Cosmology class operations.
    
    This exception is raised for issues such as:
    - Invalid cosmology preset names.
    - Failures during CAMB calculations.
    - Errors in processing CAMB results.
    - I/O errors when writing transfer functions.
    """
    pass


class Cosmology:
    """
    Handles cosmological calculations using the CAMB library.

    This class provides an interface to CAMB for computing cosmological observables
    like the matter power spectrum P(k) and the transfer function T(k). It can be
    initialized with predefined cosmological parameter sets or with custom parameters.

    Key Attributes:
        cosmoname (str): The name of the cosmology preset used or 'custom' if overridden.
        params (Dict[str, Any]): A dictionary holding all cosmological parameters,
                                 both input and derived (e.g., ombh2, omch2, sigma8_final).
        cambpars (camb.CAMBparams): The CAMB parameters object configured for the cosmology.
        camb_results (camb.results.CAMBdata): The raw results object from the CAMB calculation.
        k (np.ndarray): Array of wavenumbers (k) in h/Mpc for which P(k) and T(k) are computed.
        pk (np.ndarray): Array of linear matter power spectrum P(k) values at z=0, in (Mpc/h)^3.
                         This P(k) is normalized to `params['sigma8']` if `sigma8` was an input.
        T (np.ndarray): Array of transfer function T(k) values at z=0. 
                        Defined such that P(k) = As * (k/k_pivot_h)^ns * T(k)^2, where As is the
                        primordial scalar amplitude and k_pivot_h is the pivot scale in h/Mpc.
                        The units depend on the normalization of As. If As is dimensionless,
                        and P(k) is in (Mpc/h)^3, then T(k) has units of (Mpc/h)^(3/2).
        cosmosummary (str): A multi-line string summarizing the cosmological parameters
                            used by CAMB and key derived values. Suitable for file headers.

    Internal CAMB settings:
        _k_pivot_h_mpc (float): Pivot scale k0 used by CAMB, converted to h/Mpc.
        _kmin_h_mpc (float): Minimum k value (in h/Mpc) for P(k) calculation.
        _kmax_h_mpc (float): Maximum k value (in h/Mpc) for P(k) calculation.
        _pk_npoints (int): Number of points for P(k) calculation.
    """

    def __init__(self, cosmology: str = DEFAULT_COSMOLOGY_NAME, **kwargs: Any) -> None:
        """
        Initializes the Cosmology object, sets up CAMB, and computes cosmological parameters.

        This involves:
        1. Loading base parameters from a named preset (e.g., 'euclid-flagship').
        2. Overriding any parameters with values provided in `kwargs`.
        3. Ensuring consistency between parameters (e.g., Omega_m, Omega_b, h, ombh2, omch2).
        4. Configuring a `camb.CAMBparams` object.
        5. Running CAMB to compute power spectra and other cosmological data.
        6. Processing the results to store P(k), T(k), and derived sigma8.
        7. Generating a summary string of the cosmology.
        8. Setting up interpolation grids for z <-> chi conversions.

        Args:
            cosmology (str): The name of the cosmology preset to use, as defined in
                             `config.COSMOLOGY_PRESETS`. Defaults to `DEFAULT_COSMOLOGY_NAME`.
            **kwargs (Any): Additional cosmological parameters to override the preset values
                            or to define a custom cosmology. These should generally match
                            keys in `COSMOLOGY_PRESETS` (e.g., 'h', 'omegam', 'ombh2', 'As', 'ns', 'sigma8')
                            or standard CAMB parameter names. If 'sigma8' is provided and non-zero,
                            the resulting P(k) will be normalized to this value.

        Raises:
            CosmologyError: If the specified cosmology preset is not found, if CAMB fails
                            during calculation, or if any other critical error occurs during
                            initialization.
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
        Loads cosmology parameters from a preset and applies any specified overrides.

        It ensures that derived parameters like `ombh2` and `omch2` are consistent
        with `omegab`, `omegam`, and `h` if any of these are overridden.
        If `mnu` (total neutrino mass) is set but `numnu` (number of massive neutrino species)
        is not, `numnu` defaults to 1.

        Args:
            preset_name (str): The name of the cosmology preset to load from `COSMOLOGY_PRESETS`.
                               If not found, a warning is printed, and the default preset is used.
            **overrides (Any): Keyword arguments representing parameters to override in the preset.

        Returns:
            Dict[str, Any]: A dictionary of the final cosmological parameters.

        Raises:
            CosmologyError: If the default cosmology preset is also not found when trying to fall back.
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
        """
        Configures a `camb.CAMBparams` object based on `self.params`.

        This method translates parameters from `self.params` (which might use conventions
        like 'omegam') into the specific parameter names and structures required by CAMB
        (e.g., 'omch2', 'ombh2'). It sets cosmological parameters (H0, ombh2, omch2, omk, tau, TCMB, mnu),
        initial power spectrum parameters (As, ns), and ensures non-linear corrections are turned off.
        It handles compatibility for neutrino parameter naming across different CAMB versions where possible.
        The configured object is stored in `self.cambpars`.
        """
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
        """
        Runs the CAMB calculation to compute cosmological data.

        This method calls `self.cambpars.set_matter_power()` to specify the redshifts (z=0)
        and k-range for which the matter power spectrum should be computed. It then calls
        `camb.get_results()` to perform the actual computation. The results are stored
        in `self.camb_results`.
        It explicitly requests accurate massive neutrino transfers.
        """
        self.cambpars.set_matter_power(redshifts=[0.0], kmax=self._kmax_h_mpc, accurate_massive_neutrino_transfers=True)
        self.camb_results = camb.get_results(self.cambpars)

    def _process_camb_results(self) -> None:
        """
        Extracts and processes key results from the `self.camb_results` object.

        This method:
        1. Retrieves the matter power spectrum P(k) and corresponding k values.
        2. If `self.params['sigma8']` was provided and is non-zero, it normalizes
           the calculated P(k) to match this target sigma8. The original CAMB-calculated
           sigma8 and the normalization factor are stored in `self.params`.
        3. If `sigma8` was not provided as input, `self.params['sigma8']` is set to the
           CAMB-calculated value.
        4. Calculates the transfer function T(k) from the (potentially normalized) P(k)
           and the primordial power spectrum parameters (As, ns, k_pivot).
        5. Stores the final k, P(k), and T(k) arrays as `self.k`, `self.pk`, and `self.T`.
        6. Stores derived Omega_lambda, Omega_matter (total), and the final sigma8 (after any
           normalization) in `self.params`.
        """
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
        """
        Sets up internal grids for redshift (z) and comoving radial distance (chi)
        to enable fast interpolation for `z2chi` and `chi2z` conversions.

        A logarithmically spaced redshift grid (`self._zgrid`) is created up to `z_max_grid`.
        The corresponding comoving distances (`self._chigrid`) are calculated using
        `self.camb_results.comoving_radial_distance()`.

        Args:
            nz_grid (int): The number of points in the redshift grid. Default is 10000.
            z_max_grid (float): The maximum redshift for the grid. Default is 1100.0.

        Raises:
            CosmologyError: If `self.camb_results` is not available (i.e., CAMB hasn't run).
        """
        self._zgrid = np.logspace(-5, np.log10(z_max_grid), num=nz_grid)
        if not hasattr(self, 'camb_results'):
            raise CosmologyError("CAMB results not available for setting up interpolation grids.")
        self._chigrid = self.camb_results.comoving_radial_distance(self._zgrid)

    def z2chi(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Converts redshift(s) to comoving radial distance(s) in Mpc.

        Uses linear interpolation based on the pre-computed grids from `_setup_interpolation_grids`.
        If the grids are not yet set up, `_setup_interpolation_grids` is called first.

        Args:
            z (Union[float, np.ndarray]): A single redshift or a NumPy array of redshifts.

        Returns:
            Union[float, np.ndarray]: The corresponding comoving radial distance(s) in Mpc.
        """
        if not hasattr(self, '_zgrid') or not hasattr(self, '_chigrid'):
            self._setup_interpolation_grids()
        return np.interp(z, self._zgrid, self._chigrid)

    def chi2z(self, chi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Converts comoving radial distance(s) in Mpc to redshift(s).

        Uses linear interpolation based on the pre-computed grids from `_setup_interpolation_grids`.
        The interpolation is done from chi to z. If the grids are not yet set up,
        `_setup_interpolation_grids` is called first.

        Args:
            chi (Union[float, np.ndarray]): A single comoving radial distance in Mpc
                                           or a NumPy array of distances.

        Returns:
            Union[float, np.ndarray]: The corresponding redshift(s).
        """
        if not hasattr(self, '_zgrid') or not hasattr(self, '_chigrid'):
            self._setup_interpolation_grids()
        return np.interp(chi, self._chigrid, self._zgrid)

    def _generate_cosmosummary(self) -> None:
        """
        Generates a multi-line string summarizing the cosmological parameters used by CAMB
        and key derived values.

        This summary includes input parameters passed to CAMB (like H0, ombh2, omch2, As, ns)
        and derived parameters from CAMB results (like Omega_baryon, Omega_cdm, Omega_matter_total,
        Omega_DE, sigma8). It also includes information about sigma8 normalization if applied.
        The summary is stored in `self.cosmosummary` and is intended for inclusion in
        the header of output files, such as the transfer function file.
        If CAMB results are not available, a placeholder message is set.
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
        Writes the calculated transfer function T(k) and corresponding k values to a file.

        The output file is plain text with two columns:
        1. k (wavenumber) in h/Mpc.
        2. T(k) (transfer function), dimensionless if P(k) is (Mpc/h)^3 and As is also (Mpc/h)^3,
           or (Mpc/h)^(3/2) if As is dimensionless.

        The file includes a header section containing the detailed cosmology summary
        generated by `_generate_cosmosummary`, followed by a description of the columns.

        Args:
            filename (str): The path to the file where the transfer function will be saved.

        Raises:
            CosmologyError: If the transfer function T(k) has not been calculated yet
                            (i.e., if CAMB computations haven't successfully completed).
            IOError: If writing to the specified file fails for any reason (e.g., permissions).
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
