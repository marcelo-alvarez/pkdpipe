"""
File discovery utilities for pkdpipe.

Functions for finding simulation data files in campaign directory structures.
"""

from pathlib import Path
from typing import List, Optional


def find_simulation_data(campaign_dir: str, variant_name: str) -> Path:
    """
    Find the final snapshot file for a given simulation variant.
    
    Args:
        campaign_dir: Path to the campaign directory
        variant_name: Name of the simulation variant to find
        
    Returns:
        Path to the final snapshot file
        
    Raises:
        FileNotFoundError: If simulation directory or snapshot files not found
    """
    variant_dir = Path(campaign_dir) / "runs" / variant_name
    
    if not variant_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {variant_dir}")
    
    print(f"Searching for simulation data in: {variant_dir}")
    
    # Look for different types of snapshot files
    snapshot_patterns = ["*.tps", "*.lcp", "*.fof"]  # Tipsy, lightcone particles, halos
    snapshot_files = []
    
    # First try with extensions
    for pattern in snapshot_patterns:
        files = list(variant_dir.glob(pattern))
        if files:
            snapshot_files.extend(files)
            print(f"Found {len(files)} {pattern} files")
    
    # If no files with extensions found, look in output directory for numbered files
    if not snapshot_files:
        output_dir = variant_dir / "output"
        if output_dir.exists():
            print(f"Looking for snapshot files in: {output_dir}")
            # Look for files like variant.00001, variant.00002, etc.
            # Exclude auxiliary files (.fofstats, .hpb, .lcp, etc.)
            all_files = list(output_dir.glob(f"{variant_name}.[0-9]*"))
            files = [f for f in all_files if not any(suffix in f.name for suffix in ['.fofstats', '.hpb', '.lcp'])]
            if files:
                snapshot_files.extend(files)
                print(f"Found {len(files)} snapshot files without extensions")
    
    if not snapshot_files:
        raise FileNotFoundError(f"No simulation snapshot files found in {variant_dir}")
    
    # Sort by modification time to get the most recent (likely final) snapshot
    snapshot_files.sort(key=lambda x: x.stat().st_mtime)
    final_snapshot = snapshot_files[-1]
    
    print(f"Selected final snapshot: {final_snapshot.name}")
    print(f"File size: {final_snapshot.stat().st_size / (1024**3):.2f} GB")
    
    return final_snapshot