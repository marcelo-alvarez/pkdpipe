#!/usr/bin/env python3
"""
Generate strategic test data for pkdpipe I/O testing.

This utility script creates small test data files with strategically positioned particles
for comprehensive testing of spatial culling, multiprocessing I/O, and different
file formats (lcp, tps, fof).

USAGE:
    cd tests/
    python utils/generate_test_data.py

This script only needs to be run if:
- Test data files are corrupted or missing
- You need to modify the spatial distribution of test particles  
- You want to add new test cases or file formats
- You need to debug binary format issues

The generated test data is committed to git and should not change frequently.
"""

import numpy as np
import struct
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Define test data specifications based on pkdpipe/dataspecs.py
LIGHTCONE_DTYPE = [('mass','d'),("x",'f4'),("y",'f4'),("z",'f4'),
                   ("vx",'f4'),("vy",'f4'),("vz",'f4'),("eps",'f4'),("phi",'f4')]

TIPSY_DTYPE = [('mass','>f4'),('x','>f4'),('y','>f4'),('z','>f4'),
               ('vx','>f4'),('vy','>f4'),('vz','>f4'),('eps','>f4'),('phi','>f4')]

FOF_DTYPE = [('x','f4'), ('y','f4'), ('z','f4'),('pot','f4'),('dum1',('f4',3)),
             ('vx','f4'),('vy','f4'),('vz','f4'),('dum2',('f4',21)),
             ('npart','i4'),('dum3',('f4',2))]


def create_strategic_particles(n_particles: int, file_type: str, file_index: int) -> np.ndarray:
    """Create particles with strategic positions for testing."""
    
    if file_type == 'lightcone':
        # Lightcone particles: distributed at different distances and angles
        # for multi-file healpix testing
        
        # Create particles at different comoving distances (Gpc)
        distances = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        n_per_distance = max(1, n_particles // len(distances))
        
        particles = []
        particle_id = 0
        
        for dist in distances:
            if particle_id >= n_particles:
                break
                
            for i in range(min(n_per_distance, n_particles - particle_id)):
                # Distribute across different angular positions based on file_index
                # This ensures different healpix files get different sky coverage
                theta = (file_index * np.pi/4) + (i * np.pi/8)  # Polar angle
                phi = (file_index * np.pi/2) + (i * np.pi/4)    # Azimuthal angle
                
                # Convert spherical to cartesian (lightcone coordinates)
                x = dist * np.cos(phi) * np.sin(theta)
                y = dist * np.sin(phi) * np.sin(theta) 
                z = dist * np.cos(theta)
                
                # Add some velocity (peculiar motion)
                vx = 100 * np.random.normal()  # km/s
                vy = 100 * np.random.normal()
                vz = 100 * np.random.normal()
                
                particles.append((
                    1.0e10,  # mass (solar masses)
                    x, y, z,
                    vx, vy, vz,
                    0.01,    # eps (softening)
                    -1.0e5   # phi (potential)
                ))
                particle_id += 1
        
        return np.array(particles, dtype=LIGHTCONE_DTYPE)
        
    elif file_type == 'snapshot':
        # Snapshot particles: distributed in cubic regions for bbox testing
        # Different files cover different spatial regions
        
        box_size = 1000.0  # Mpc/h
        region_size = box_size / 4  # Each file covers 1/4 of box in each dimension
        
        # File index determines which spatial region
        x_offset = (file_index % 2) * region_size - box_size/2
        y_offset = ((file_index // 2) % 2) * region_size - box_size/2
        z_offset = (file_index // 4) * region_size - box_size/2
        
        particles = []
        for i in range(n_particles):
            # Create particles within this file's spatial region
            x = x_offset + np.random.uniform(0, region_size)
            y = y_offset + np.random.uniform(0, region_size)
            z = z_offset + np.random.uniform(0, region_size)
            
            # Add some particles exactly on bbox boundaries for edge case testing
            if i < 3:
                if i == 0: x = x_offset  # On boundary
                if i == 1: y = y_offset + region_size  # On opposite boundary
                if i == 2: z = z_offset + region_size/2  # In middle
            
            # Velocities based on position (simple Hubble flow + peculiar)
            H0 = 70.0  # km/s/Mpc
            vx = H0 * x / 1000 + 200 * np.random.normal()
            vy = H0 * y / 1000 + 200 * np.random.normal()
            vz = H0 * z / 1000 + 200 * np.random.normal()
            
            particles.append((
                1.0e10,  # mass
                x, y, z,
                vx, vy, vz,
                0.01,    # eps
                -1.0e5   # phi
            ))
        
        return np.array(particles, dtype=TIPSY_DTYPE)
        
    elif file_type == 'halo':
        # Halo data: different mass halos at strategic positions
        
        box_size = 1000.0
        particles = []
        
        # Create halos with different particle counts (masses)
        halo_masses = [10, 50, 100, 500, 1000, 5000]  # Number of particles
        n_per_mass = max(1, n_particles // len(halo_masses))
        
        particle_id = 0
        for mass in halo_masses:
            if particle_id >= n_particles:
                break
                
            for i in range(min(n_per_mass, n_particles - particle_id)):
                # Distribute halos across box, with some bias by file_index
                x = (file_index * box_size/4) + np.random.uniform(-box_size/2, box_size/2)
                y = np.random.uniform(-box_size/2, box_size/2)
                z = np.random.uniform(-box_size/2, box_size/2)
                
                # Halo velocities (bulk motion)
                vx = 300 * np.random.normal()
                vy = 300 * np.random.normal()
                vz = 300 * np.random.normal()
                
                particles.append((
                    x, y, z,
                    -1.0e6,  # potential
                    [0.0, 0.0, 0.0],  # dum1 (3 floats)
                    vx, vy, vz,
                    [0.0] * 21,  # dum2 (21 floats)
                    mass,    # number of particles in halo
                    [0.0, 0.0]  # dum3 (2 floats)
                ))
                particle_id += 1
        
        return np.array(particles, dtype=FOF_DTYPE)


def write_lightcone_file(filepath: Path, particles: np.ndarray):
    """Write lightcone format file (.hpb)."""
    # Lightcone format: 40 bytes per particle, no header
    with open(filepath, 'wb') as f:
        for particle in particles:
            # Pack according to lightcone format
            data = struct.pack('d6f2f', 
                              particle['mass'],
                              particle['x'], particle['y'], particle['z'],
                              particle['vx'], particle['vy'], particle['vz'],
                              particle['eps'], particle['phi'])
            f.write(data)


def write_tipsy_file(filepath: Path, particles: np.ndarray):
    """Write tipsy format file with header."""
    # Tipsy format: 32-byte header + 36 bytes per particle
    n_particles = len(particles)
    
    with open(filepath, 'wb') as f:
        # Write tipsy header (32 bytes total)
        # Standard tipsy header: time(8) + ntotal(4) + ndim(4) + nsph(4) + ndark(4) + nstar(4) + pad(4)
        header = struct.pack('>d6i', 
                           0.0,         # time (8 bytes)
                           n_particles, # ntotal (4 bytes)
                           3,           # ndim (4 bytes) 
                           0,           # nsph (4 bytes)
                           n_particles, # ndark (4 bytes) - all our particles are dark matter
                           0,           # nstar (4 bytes)
                           0)           # padding (4 bytes)
        f.write(header)
        
        # Write particle data (big-endian format)
        for particle in particles:
            data = struct.pack('>9f',
                              particle['mass'],
                              particle['x'], particle['y'], particle['z'],
                              particle['vx'], particle['vy'], particle['vz'],
                              particle['eps'], particle['phi'])
            f.write(data)


def write_fof_file(filepath: Path, particles: np.ndarray):
    """Write friends-of-friends halo format file."""
    # FOF format: 132 bytes per halo
    with open(filepath, 'wb') as f:
        for particle in particles:
            # Pack according to FOF format: x,y,z,pot,dum1(3),vx,vy,vz,dum2(21),npart,dum3(2)
            data = struct.pack('4f3f3f21fi2f',
                              particle['x'], particle['y'], particle['z'], particle['pot'],
                              *particle['dum1'],  # 3 floats
                              particle['vx'], particle['vy'], particle['vz'],
                              *particle['dum2'],  # 21 floats
                              particle['npart'],
                              *particle['dum3'])  # 2 floats
            f.write(data)


def create_test_parameter_file(test_data_dir: Path) -> Dict[str, Any]:
    """Create a test parameter file pointing to our test data."""
    
    param_content = f'''# Test parameter file for pkdpipe I/O testing
achOutName      = "{test_data_dir}/test_sim"
achTfFile       = "{test_data_dir}/test_transfer.dat"

# Lightcone
bLightCone          = 1
bLightConeParticles = 1
dRedshiftLCP        = 0.5
nSideHealpix       = 4    # Small for testing
sqdegLCP           = -1
hLCP               = [1, 0, 0]

# Simulation box
dBoxSize        = 1000    # Mpc/h
nGrid           = 128     # Small for testing
iLPT            = 2
iSeed           = 314159
dRedFrom        = 5
bWriteIC        = False

# Cosmology  
h               = 0.7
dOmega0         = 0.3
dLambda         = 0.7
dSigma8         = 0.8
dSpectral       = 0.96

iStartStep      = 0
nSteps          = [1]
dRedTo          = 0.0
bLightCone      = 1
'''
    
    param_file = test_data_dir / "test_sim.par"
    with open(param_file, 'w') as f:
        f.write(param_content)
    
    # Create minimal transfer function file
    transfer_file = test_data_dir / "test_transfer.dat"
    with open(transfer_file, 'w') as f:
        f.write("# k [h/Mpc]  T(k)\n")
        f.write("0.001  1.0\n")
        f.write("0.1    0.5\n")
        f.write("1.0    0.1\n")
    
    return {
        'param_file': str(param_file),
        'transfer_file': str(transfer_file),
        'box_size': 1000.0,
        'ngrid': 128
    }


def generate_all_test_data():
    """Generate complete test data suite."""
    
    # Setup directories
    test_data_dir = Path(__file__).parent / "test_data"
    lightcone_dir = test_data_dir / "lightcone"
    snapshots_dir = test_data_dir / "snapshots" 
    halos_dir = test_data_dir / "halos"
    
    # Create parameter file
    param_info = create_test_parameter_file(test_data_dir)
    
    # Generate lightcone files (4 healpix pixels)
    print("Generating lightcone test data...")
    lightcone_particles_per_file = 20
    for i in range(4):
        particles = create_strategic_particles(lightcone_particles_per_file, 'lightcone', i)
        filepath = lightcone_dir / f"test_sim.00001.hpb.{i}"
        write_lightcone_file(filepath, particles)
        print(f"  Created {filepath} with {len(particles)} particles")
    
    # Generate snapshot files (3 process files)  
    print("Generating snapshot test data...")
    snapshot_particles_per_file = 15
    for i in range(3):
        particles = create_strategic_particles(snapshot_particles_per_file, 'snapshot', i)
        filepath = snapshots_dir / f"test_sim.00001.{i}"
        write_tipsy_file(filepath, particles)
        print(f"  Created {filepath} with {len(particles)} particles")
    
    # Generate halo files (3 process files)
    print("Generating halo test data...")
    halos_per_file = 10
    for i in range(3):
        halos = create_strategic_particles(halos_per_file, 'halo', i)
        filepath = halos_dir / f"test_sim.00001.fofstats.{i}"
        write_fof_file(filepath, halos)
        print(f"  Created {filepath} with {len(halos)} halos")
    
    # Generate expected results for validation
    expected_results = {
        'lightcone': {
            'total_files': 4,
            'particles_per_file': lightcone_particles_per_file,
            'file_pattern': 'test_sim.00001.hpb.*',
            'distance_range': [0.1, 4.0],  # Gpc
            'test_bbox': [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]  # Should contain most particles
        },
        'snapshots': {
            'total_files': 3,
            'particles_per_file': snapshot_particles_per_file,
            'file_pattern': 'test_sim.00001.*',
            'box_size': 1000.0,  # Mpc/h
            'test_bbox': [[-200, 200], [-200, 200], [-200, 200]]  # Should contain some particles
        },
        'halos': {
            'total_files': 3,
            'halos_per_file': halos_per_file,
            'file_pattern': 'test_sim.00001.fofstats.*',
            'mass_range': [10, 5000],  # particle counts
            'test_bbox': [[-300, 300], [-300, 300], [-300, 300]]
        },
        'parameter_info': param_info
    }
    
    # Save expected results
    results_file = test_data_dir / "expected_results.json"
    with open(results_file, 'w') as f:
        json.dump(expected_results, f, indent=2)
    
    print(f"\nTest data generation complete!")
    print(f"Generated files in: {test_data_dir}")
    print(f"Expected results saved to: {results_file}")
    
    return expected_results


if __name__ == "__main__":
    generate_all_test_data()