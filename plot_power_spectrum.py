#!/usr/bin/env python3
"""
Plot Power Spectrum with Chi-Squared Error Bars

This script reads the power spectrum data from power_spectrum_ngrid256_cic.txt
and creates a plot with error bars based on the N_modes column using chi-squared statistics.

For power spectrum measurements, the error on P(k) follows chi-squared statistics:
σ_P(k) = P(k) * sqrt(2/N_modes)

where N_modes is the number of Fourier modes contributing to each k-bin.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def load_power_spectrum(filename):
    """Load power spectrum data from file."""
    try:
        # Load only the numeric columns, skip the status column
        data = np.loadtxt(filename, usecols=(0, 1, 2))
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Make sure the power spectrum calculation has completed successfully.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def calculate_chi_squared_errors(power_spectrum, n_modes):
    """
    Calculate chi-squared error bars for power spectrum measurements.
    
    For power spectrum measurements, the standard error follows:
    σ_P(k) = P(k) * sqrt(2/N_modes)
    
    This comes from the fact that power spectrum measurements follow
    chi-squared statistics with N_modes degrees of freedom.
    """
    # Avoid division by zero
    n_modes = np.maximum(n_modes, 1)
    
    # Chi-squared error formula for power spectrum
    errors = power_spectrum * np.sqrt(2.0 / n_modes)
    
    return errors

def create_power_spectrum_plot(k_values, power_spectrum, n_modes, filename_base="power_spectrum"):
    """Create a publication-quality power spectrum plot with error bars."""
    
    # Calculate error bars
    errors = calculate_chi_squared_errors(power_spectrum, n_modes)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot power spectrum with error bars
    # plt.errorbar(k_values, power_spectrum, yerr=errors, 
    #             fmt='o-', markersize=6, capsize=4, capthick=2,
    #             linewidth=2, label='Power Spectrum P(k)')
    plt.plot(k_values, power_spectrum, linewidth=2, label='Power Spectrum P(k)')    
    # Set logarithmic scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Labels and title
    plt.xlabel('k [h/Mpc]', fontsize=14)
    plt.ylabel('P(k) [(Mpc/h)³]', fontsize=14)
    plt.title('Matter Power Spectrum (256³ Grid, CIC Assignment)', fontsize=16)
    
    # Grid for better readability
    plt.grid(True, alpha=0.3, which='both')
    
    # Legend
    plt.legend(fontsize=12)
    
    # Add text box with statistics
    total_modes = np.sum(n_modes)
    k_range = f"k ∈ [{k_values.min():.3f}, {k_values.max():.3f}] h/Mpc"
    stats_text = f"Total modes: {total_modes:,}\n{k_range}\nBins: {len(k_values)}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    output_file = f"{filename_base}_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Power spectrum plot saved to: {output_file}")
    
    # Also save as PDF for publications
    output_pdf = f"{filename_base}_plot.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"✅ Power spectrum plot saved to: {output_pdf}")
    
    # Show the plot
    plt.show()
    
    return output_file, output_pdf

def print_statistics(k_values, power_spectrum, n_modes, errors):
    """Print summary statistics of the power spectrum."""
    print("\n" + "="*60)
    print("POWER SPECTRUM ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Number of k-bins: {len(k_values)}")
    print(f"k-range: {k_values.min():.6f} to {k_values.max():.6f} h/Mpc")
    print(f"Power spectrum range: {power_spectrum.min():.2e} to {power_spectrum.max():.2e} (Mpc/h)³")
    print(f"Total modes: {np.sum(n_modes):,}")
    print(f"Average modes per bin: {np.mean(n_modes):.0f}")
    print(f"Median relative error: {np.median(errors/power_spectrum)*100:.1f}%")
    
    # Find bins with good statistics (>100 modes)
    good_bins = n_modes > 100
    print(f"Bins with >100 modes: {np.sum(good_bins)}/{len(k_values)}")
    
    if np.any(good_bins):
        print(f"Best measured k-range: {k_values[good_bins].min():.6f} to {k_values[good_bins].max():.6f} h/Mpc")
    
    print("="*60)

def main():
    """Main function."""
    # Default filename
    filename = "power_spectrum_ngrid256_cic.txt"
    
    # Check if file exists
    if not Path(filename).exists():
        print(f"Looking for power spectrum file: {filename}")
        print("File not found. Please ensure the power spectrum calculation has completed.")
        sys.exit(1)
    
    print(f"📊 Loading power spectrum data from: {filename}")
    
    # Load the data
    # Expected format: k_values, power_spectrum, n_modes
    data = load_power_spectrum(filename)
    
    if data.shape[1] < 3:
        print(f"Error: Expected at least 3 columns (k, P(k), N_modes), got {data.shape[1]}")
        print("File format should be: k_values  power_spectrum  n_modes")
        sys.exit(1)
    
    # Extract columns
    k_values = data[:, 0]
    power_spectrum = data[:, 1] 
    n_modes = data[:, 2]
    
    print(f"✅ Loaded {len(k_values)} k-bins")
    
    # Calculate error bars
    errors = calculate_chi_squared_errors(power_spectrum, n_modes)
    
    # Print statistics
    print_statistics(k_values, power_spectrum, n_modes, errors)
    
    # Create the plot
    print("\n📈 Creating power spectrum plot...")
    filename_base = filename.replace('.txt', '')
    
    plot_files = create_power_spectrum_plot(k_values, power_spectrum, n_modes, filename_base)
    
    print("\n✅ Power spectrum plotting completed!")
    print(f"Plot files created: {plot_files[0]}, {plot_files[1]}")

if __name__ == "__main__":
    main()