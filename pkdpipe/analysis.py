"""
Analysis utilities for pkdpipe.

Functions for analyzing and validating simulation results, particularly
power spectrum analysis and statistical validation.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


def analyze_results(k_bins: np.ndarray, power_spectrum: np.ndarray, 
                   n_modes: np.ndarray, density_stats: Dict[str, float],
                   box_size: float, n_particles: int, ngrid: int, 
                   assignment: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze and display power spectrum results with comprehensive validation.
    
    Args:
        k_bins: Array of k-bin centers in h/Mpc
        power_spectrum: Array of power spectrum values in (Mpc/h)³
        n_modes: Array of number of modes per k-bin
        density_stats: Dictionary containing density field statistics
        box_size: Simulation box size in Mpc/h
        n_particles: Total number of particles
        ngrid: Grid resolution used for FFT
        assignment: Mass assignment scheme used ('cic', 'ngp', etc.)
        output_file: Optional output filename for results
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    print(f"\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Frequency analysis
    cell_size = box_size / ngrid
    k_fundamental = 2 * np.pi / box_size
    k_nyquist = np.pi / cell_size
    k_cutoff = 0.5 * k_nyquist
    
    print(f"Frequency Analysis:")
    print(f"  Fundamental mode: {k_fundamental:.6f} h/Mpc")
    print(f"  Nyquist frequency: {k_nyquist:.3f} h/Mpc")
    print(f"  Recommended cutoff: {k_cutoff:.3f} h/Mpc")
    
    # Basic statistics - handle missing density stats gracefully
    print(f"\nDensity Field:")
    if 'mean_density' in density_stats and not np.isnan(density_stats['mean_density']):
        print(f"  Mean density: {density_stats['mean_density']:.6e}")
        
        # Calculate global mean density for comparison if we have particle count
        global_mean_density = n_particles / (box_size ** 3)
        print(f"  Global mean density (from particle count): {global_mean_density:.6e}")
        
        # Check consistency
        density_ratio = density_stats['mean_density'] / global_mean_density
        print(f"  Density ratio (gridded/global): {density_ratio:.4f}")
        if abs(density_ratio - 1.0) > 0.1:
            print(f"  WARNING: Large discrepancy in density calculation!")
    else:
        # Fallback: calculate from total particles and box volume
        global_mean_density = n_particles / (box_size ** 3)
        print(f"  Global mean density (from particle count): {global_mean_density:.6e}")
        print(f"  Note: Detailed density statistics not available in distributed mode")
    
    # Other density statistics if available
    if 'density_variance' in density_stats and not np.isnan(density_stats['density_variance']):
        print(f"  Density variance: {density_stats['density_variance']:.6e}")
    if 'delta_variance' in density_stats and not np.isnan(density_stats['delta_variance']):
        print(f"  Delta contrast variance: {density_stats['delta_variance']:.6e}")
    
    # Display power spectrum summary
    valid_mask = (k_bins <= k_cutoff) & (n_modes > 100)
    valid_bins = np.sum(valid_mask)
    
    print(f"\nPower Spectrum Results:")
    print(f"  Total k-bins: {len(k_bins)}")
    print(f"  Valid k-bins (within cutoff): {valid_bins}")
    print(f"  Power spectrum range: {power_spectrum.min():.2e} to {power_spectrum.max():.2e} (Mpc/h)³")
    print(f"  Total modes measured: {n_modes.sum():,}")
    
    # Prepare results dictionary
    analysis_results = {
        'frequency_analysis': {
            'k_fundamental': k_fundamental,
            'k_nyquist': k_nyquist,
            'k_cutoff': k_cutoff,
            'cell_size': cell_size
        },
        'density_analysis': {
            'global_mean_density': n_particles / (box_size ** 3),
            'density_stats': density_stats
        },
        'power_spectrum_summary': {
            'total_bins': len(k_bins),
            'valid_bins': valid_bins,
            'power_range': (power_spectrum.min(), power_spectrum.max()),
            'total_modes': n_modes.sum()
        },
        'parameters': {
            'box_size': box_size,
            'ngrid': ngrid,
            'assignment': assignment,
            'n_particles': n_particles
        }
    }
    
    # Save power spectrum to file if requested
    if output_file is None:
        output_file = f"power_spectrum_ngrid{ngrid}_{assignment}.txt"
    
    print(f"\nSaving power spectrum to: {output_file}")
    
    # Create header with metadata
    header = f"""# Power Spectrum Analysis Results
# Grid size: {ngrid}³
# Box size: {box_size:.1f} Mpc/h
# Assignment: {assignment.upper()}
# Cell size: {cell_size:.6f} Mpc/h
# Fundamental mode: {k_fundamental:.6f} h/Mpc
# Nyquist frequency: {k_nyquist:.6f} h/Mpc
# Cutoff frequency: {k_cutoff:.6f} h/Mpc
# Total particles: {n_particles:,}
# Total modes: {n_modes.sum():,}
# Valid k-bins: {valid_bins}
#
# Columns: k[h/Mpc] P(k)[(Mpc/h)³] N_modes Status
"""
    
    with open(output_file, 'w') as f:
        f.write(header)
        for i in range(len(k_bins)):
            if n_modes[i] > 0:
                status = "valid" if k_bins[i] <= k_cutoff else "aliased"
                f.write(f"{k_bins[i]:.6f} {power_spectrum[i]:.6e} {n_modes[i]} {status}\n")
    
    print(f"✅ Power spectrum analysis completed successfully!")
    print(f"✅ Results saved to {output_file}")
    
    analysis_results['output_file'] = output_file
    return analysis_results


def validate_power_spectrum(k_bins: np.ndarray, power_spectrum: np.ndarray, 
                          n_modes: np.ndarray) -> Dict[str, bool]:
    """
    Validate power spectrum results for common issues.
    
    Args:
        k_bins: Array of k-bin centers
        power_spectrum: Array of power spectrum values  
        n_modes: Array of number of modes per k-bin
        
    Returns:
        Dictionary of validation results
    """
    validation = {
        'positive_power': np.all(power_spectrum > 0),
        'monotonic_k': np.all(np.diff(k_bins) > 0),
        'positive_modes': np.all(n_modes >= 0),
        'reasonable_range': (power_spectrum.min() > 1e-10) and (power_spectrum.max() < 1e10)
    }
    
    validation['all_passed'] = all(validation.values())
    return validation