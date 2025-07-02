# -*- coding: utf-8 -*-
"""
PKDGrav3 data interface - Refactored for better maintainability and robustness.

MULTI-LEVEL PARALLEL I/O ARCHITECTURE
=====================================

This module implements a sophisticated multi-level parallel I/O system designed for 
large-scale distributed computing environments, particularly SLURM-managed GPU clusters.

Architecture Overview:
---------------------

Level 1: SLURM Process Distribution
- One SLURM process per GPU (typically via srun)
- Each process identified by SLURM_PROCID (0 to SLURM_NTASKS-1)
- Each process has SLURM_CPUS_PER_TASK CPU cores available

Level 2: Local Multiprocessing (within each SLURM process)
- Uses Python multiprocessing.Pool when beneficial
- Pool size = min(available_files_per_slurm_process, SLURM_CPUS_PER_TASK)

Adaptive Strategy by File Count:
-------------------------------

1. MANY FILES (> SLURM_NTASKS):
   - Distribute files across SLURM processes
   - Each SLURM process uses multiprocessing.Pool for its assigned files
   - Maximum concurrency: SLURM_NTASKS × SLURM_CPUS_PER_TASK
   - Example: 32 SLURM processes × 32 cores = 1024 concurrent file reads

2. FEW FILES (≤ SLURM_NTASKS):
   - Assign one complete file per SLURM process
   - No multiprocessing needed within SLURM processes
   - Maximum concurrency: number_of_files

3. SINGLE FILE:
   - SLURM-level chunking: each SLURM process reads one chunk
   - Uses chunk % SLURM_NTASKS == SLURM_PROCID for assignment
   - No multiprocessing within SLURM processes (avoids JAX conflicts)
   - Chunk size: typically 2GB for TPS format

JAX Compatibility:
-----------------
- Single-file chunking avoids multiprocessing.Pool to prevent JAX deadlocks
- Multi-file scenarios complete I/O before JAX initialization
- SLURM-level distribution is process-safe with JAX

Environment Variables:
---------------------
- SLURM_NTASKS: Total number of SLURM processes
- SLURM_PROCID: Current SLURM process ID (0-indexed)
- SLURM_CPUS_PER_TASK: CPU cores available per SLURM process

Example Configurations:
----------------------
8 nodes × 4 GPUs/node × 32 cores/GPU:
- SLURM_NTASKS = 32
- SLURM_CPUS_PER_TASK = 32
- Max concurrency = 1024 operations

4 nodes × 2 GPUs/node × 16 cores/GPU:
- SLURM_NTASKS = 8  
- SLURM_CPUS_PER_TASK = 16
- Max concurrency = 128 operations
"""
import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d
import os
import re
import logging
import sys
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import pkdpipe.dataspecs as pkds  
from pkdpipe.cosmology import Cosmology


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_master_process():
    """
    Check if this is the master process in a distributed environment.
    
    This function helps suppress duplicate logging from worker processes in
    multi-process environments like SLURM jobs. JAX-specific process detection
    is avoided here to prevent premature JAX initialization that can cause
    fork warnings when JAX is later properly initialized for distributed computing.
    
    Returns:
        True if this is the master process or if environment can't be determined
    """
    try:
        # Check if we're in a SLURM multi-process environment
        slurm_procid = os.environ.get('SLURM_PROCID')
        if slurm_procid is not None:
            return int(slurm_procid) == 0
        
        # Check JAX distributed environment variables without importing JAX
        # This avoids premature JAX initialization that causes fork warnings
        jax_process_id = os.environ.get('JAX_PROCESS_ID')
        if jax_process_id is not None:
            return int(jax_process_id) == 0
            
        # Default to master if we can't determine
        return True
    except Exception:
        return True


def _log_info(message):
    """
    Log info message only from master process.
    
    This prevents duplicate log messages when running in distributed/multi-process 
    environments (e.g., SLURM jobs with multiple tasks). Warning and error messages
    are still logged from all processes as they may indicate process-specific issues.
    
    Args:
        message: The message to log
    """
    if _is_master_process():
        logger.info(message)


def _get_slurm_info():
    """
    Get SLURM environment information for distributed I/O.
    
    Returns:
        Dict with keys:
        - 'ntasks': Total number of SLURM processes 
        - 'procid': Current SLURM process ID (0-indexed)
        - 'cpus_per_task': CPU cores per SLURM process
        - 'is_slurm': Whether running in SLURM environment
    """
    try:
        ntasks = int(os.environ.get('SLURM_NTASKS', '1'))
        procid = int(os.environ.get('SLURM_PROCID', '0'))
        cpus_per_task = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
        is_slurm = 'SLURM_NTASKS' in os.environ
        
        return {
            'ntasks': ntasks,
            'procid': procid, 
            'cpus_per_task': cpus_per_task,
            'is_slurm': is_slurm
        }
    except (ValueError, TypeError):
        # Fallback if environment variables are malformed
        return {
            'ntasks': 1,
            'procid': 0,
            'cpus_per_task': 1,
            'is_slurm': False
        }


@dataclass
class SimulationParams:
    """Container for simulation parameters."""
    namespace: str
    zmax: float
    boxsize: int
    nsteps: int
    omegam: float
    h: float
    ngrid: int
    zoutput: np.ndarray


class ParameterParser:
    """Handles parsing of PKDGrav3 parameter files."""
    
    @staticmethod
    def parse_parameter(content, varname, vartype):
        """Parse a variable from parameter file content.
        
        Args:
            content: The parameter file content as a string
            varname: Name of the parameter to find
            vartype: Expected type (str, float, int)
            
        Returns:
            Parsed value or None if not found
            
        Raises:
            ValueError: If parameter format is invalid
        """
        escaped_name = re.escape(varname)
        
        try:
            if vartype == str:
                match = re.search(rf'{escaped_name}\s*=\s*["\'"]([^"\']+)["\'"]', content)
                return match.group(1) if match else None
                
            elif vartype == float:
                match = re.search(rf'{escaped_name}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content)
                return float(match.group(1)) if match else None
                
            elif vartype == int:
                # Try list format first: nSteps = [1, 2, 3, ...]
                list_match = re.search(rf'{escaped_name}\s*=\s*\[([^\]]+)\]', content)
                if list_match:
                    list_content = list_match.group(1)
                    numbers = [int(x.strip()) for x in list_content.split(',') if x.strip()]
                    return sum(numbers)
                
                # Try single integer
                match = re.search(rf'{escaped_name}\s*=\s*([-+]?\d+)', content)
                return int(match.group(1)) if match else None
                
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Failed to parse parameter '{varname}' as {vartype.__name__}: {e}")
        
        return None

    @staticmethod
    def load_parameters(param_file):
        """Load and validate simulation parameters from file.
        
        Args:
            param_file to the parameter file
            
        Returns:
            SimulationParams object with loaded parameters
            
        Raises:
            FileNotFoundError: If parameter file doesn't exist
            ValueError: If required parameters are missing or invalid
        """
        if not param_file.exists():
            raise FileNotFoundError(f"Parameter file not found: {param_file}")
        
        try:
            content = param_file.read_text()
        except IOError as e:
            raise IOError(f"Failed to read parameter file {param_file}: {e}")
        
        # Parse required parameters
        required_params = {
            'namespace': ('achOutName', str),
            'zmax': ('dRedshiftLCP', float),
            'boxsize': ('dBoxSize', int),
            'nsteps': ('nSteps', int),
            'omegam': ('dOmega0', float),
            'h': ('h', float),
            'ngrid': ('nGrid', int),
        }
        
        params = {}
        missing_params = []
        
        for param_name, (file_param, param_type) in required_params.items():
            value = ParameterParser.parse_parameter(content, file_param, param_type)
            if value is None:
                missing_params.append(file_param)
            else:
                params[param_name] = value
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Load zoutput from log file
        log_file = Path(params['namespace'] + ".log")
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        try:
            zoutput = np.genfromtxt(log_file)[:, 1]
        except (IndexError, ValueError) as e:
            raise ValueError(f"Failed to load zoutput from {log_file}: {e}")
        
        params['zoutput'] = zoutput
        
        return SimulationParams(**params)


class DataProcessor:
    """Handles data processing operations."""
    
    @staticmethod
    def cull_shift_reshape(vars, data, shift,
                          bounds):
        """Cull, shift, and reshape data based on variables and bounds."""
        if data.size == 0:
            return np.array([]).reshape(len(vars), 0)
        
        cdata = np.copy(data)
        
        # Use all requested variables from dataset specification
        active_vars = vars
        
        # Apply shifts
        for var in ['x', 'y', 'z']:
            if var in cdata.dtype.names:
                cdata[var] += shift[var]
        
        # Apply spatial bounds
        if all(var in active_vars for var in ['x', 'y', 'z']):
            mask = ((cdata['x'] > bounds[0][0]) & (cdata['x'] <= bounds[0][1]) &
                   (cdata['y'] > bounds[1][0]) & (cdata['y'] <= bounds[1][1]) &
                   (cdata['z'] > bounds[2][0]) & (cdata['z'] <= bounds[2][1]))
            
            # Apply radial bounds if specified
            if len(bounds) > 3:
                r = np.sqrt(cdata['x']**2 + cdata['y']**2 + cdata['z']**2)
                mask = mask & (r > bounds[3][0]) & (r <= bounds[3][1])
        else:
            mask = np.ones(len(cdata), dtype=bool)
        
        # Extract and reshape selected data - only position coordinates
        try:
            # Check if any particles pass the mask
            if not np.any(mask):
                return np.array([]).reshape(len(active_vars), 0)
                
            result = np.concatenate([cdata[var][mask] for var in active_vars])
            return result.reshape((len(active_vars), -1))
        except KeyError as e:
            raise ValueError(f"Variable {e} not found in data")

    @staticmethod
    def cull_tile_reshape(data, cdata, vars, 
                         bbox, r1, r2,
                         format_info, lightcone):
        """Cull, tile, and reshape data."""
        shifts = []
        bounds = bbox.copy()
        
        if not format_info.get('sliced', True) and lightcone:
            # Create 8 corner shifts for lightcone mode
            offsets = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
            d = np.stack(np.meshgrid(*offsets, indexing='ij'), axis=-1).reshape(-1, 3)
            for offset in d:
                shifts.append(np.rec.fromarrays(offset, names=['x', 'y', 'z']))
            bounds.append([r1, r2])
        else:
            # Single shift based on format offset
            offset = -np.asarray(format_info.get('offset', [0, 0, 0]))
            shifts.append(np.rec.fromarrays(offset, names=['x', 'y', 'z']))
        
        for shift in shifts:
            sdata = DataProcessor.cull_shift_reshape(vars, cdata, shift, bounds)
            if sdata.size > 0:
                data = np.concatenate((data, sdata), axis=1)
        
        return data

    @staticmethod
    def cull_tile_reshape_single(cdata, vars, bbox, r1, r2, format_info, lightcone):
        """Cull, tile, and reshape data for a single chunk (memory efficient version)."""
        shifts = []
        bounds = bbox.copy()
        
        if not format_info.get('sliced', True) and lightcone:
            # Create 8 corner shifts for lightcone mode
            offsets = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])
            d = np.stack(np.meshgrid(*offsets, indexing='ij'), axis=-1).reshape(-1, 3)
            for offset in d:
                shifts.append(np.rec.fromarrays(offset, names=['x', 'y', 'z']))
            bounds.append([r1, r2])
        else:
            # Single shift based on format offset
            offset = -np.asarray(format_info.get('offset', [0, 0, 0]))
            shifts.append(np.rec.fromarrays(offset, names=['x', 'y', 'z']))
        
        # Collect data from all shifts
        chunk_data_list = []
        for shift in shifts:
            sdata = DataProcessor.cull_shift_reshape(vars, cdata, shift, bounds)
            if sdata.size > 0:
                chunk_data_list.append(sdata)
        
        # Concatenate all shifts for this chunk
        if chunk_data_list:
            return np.concatenate(chunk_data_list, axis=1)
        else:
            return np.array([], dtype=np.float32).reshape(len(vars), 0)


class FileReader:
    """Handles file I/O operations."""
    
    @staticmethod
    def get_filename(namespace, format_info, chunk, step):
        """Generate filename for a given chunk and step.
        
        For formats with extensions (like 'lcp'), generates:
          - namespace.00001.lcp.0, namespace.00001.lcp.1, etc.
        
        For formats without extensions (like 'tps'), generates:
          - chunk 0: namespace.00001 (primary file)
          - chunk 1+: namespace.00001.1, namespace.00001.2, etc.
        """
        ext = format_info.get('ext')
        if ext is not None:
            # Format with extension: namespace.step.ext.chunk
            filename = f"{namespace}.{step:05d}.{ext}.{chunk}"
        else:
            # Format without extension: 
            # - chunk 0: namespace.step (primary file)
            # - chunk 1+: namespace.step.chunk
            if chunk == 0:
                filename = f"{namespace}.{step:05d}"
            else:
                filename = f"{namespace}.{step:05d}.{chunk}"
        return Path(filename)
    
    @staticmethod
    def read_chunk(filepath, dtype, offset = 0, count = -1):
        """Read a chunk of data from file."""
        if not filepath.exists():
            return np.array([], dtype=dtype)
        
        try:
            return np.fromfile(filepath, dtype=dtype, offset=offset, count=count)
        except (IOError, ValueError) as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return np.array([], dtype=dtype)
    
    @staticmethod
    def find_files(namespace, step, format_info = None, max_search = 1000):
        """Find all chunk files for a given step.
        
        Args:
            namespace: Base filename namespace
            step: Time step number
            format_info: File format information (optional, defaults to no extension)
            max_search: Maximum number of chunks to search for
            
        Returns:
            List of chunk numbers that have existing files
        """
        available_files = []
        chunk = 0
        
        while chunk < max_search:
            # Use get_filename to handle both extension and no-extension formats
            if format_info is not None:
                filepath = FileReader.get_filename(namespace, format_info, chunk, step)
            else:
                # Fallback for backward compatibility (no extension format)
                if chunk == 0:
                    filepath = Path(f"{namespace}.{step:05d}")
                else:
                    filepath = Path(f"{namespace}.{step:05d}.{chunk}")
            
            if filepath.exists():
                available_files.append(chunk)
                chunk += 1
            else:
                break
        
        return available_files


class Data:
    """PKDGrav3 data interface - refactored for better maintainability."""

    def __init__(self, *, param_file, nproc = 1, verbose = False):
        """Initialize the data interface.
        
        Args:
            param_file to the PKDGrav3 parameter file
            nproc: Number of processes for parallel operations
            verbose: Enable verbose logging
        """
        self.param_file = Path(param_file)
        self.nproc = max(1, nproc)
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Load and validate parameters
        self.params = ParameterParser.load_parameters(self.param_file)
        
        # Initialize cosmology interpolation
        self._init_cosmology()
        
        _log_info(f"Initialized data interface for {self.params.namespace}")
        _log_info(f"Found {len(self.params.zoutput)} output redshifts")

    def _init_cosmology(self):
        """Initialize cosmological distance interpolation."""
        zhigh, zlow, nz = 1100.0, 0.0, 1000
        a = np.logspace(np.log10(1 / (1 + zhigh)), np.log10(1 / (1 + zlow)), nz)
        z = 1 / a - 1
        
        cosmology = Cosmology(h=self.params.h, omegam=self.params.omegam)
        chi = cosmology.z2chi(z)
        self.chiofz = interp1d(z, chi, kind='linear', fill_value='extrapolate')

    def _read_step(self, step, bbox, dprops, 
                   format_info, lightcone, redshift, 
                   chunktask):
        """Read data for a single time step."""
        zvals = self.params.zoutput
        
        # Validate step index
        if step >= len(zvals) or step < 1:
            logger.warning(f"Invalid step {step}, must be between 1 and {len(zvals)-1}")
            return np.array([], dtype=np.float32).reshape(len(dprops['vars']), 0)
        
        z1, z2 = zvals[step], zvals[step - 1]
        vars = dprops['vars']
        dtype = format_info['dtype']
        
        # Calculate distance bounds
        chi1, chi2 = self.chiofz(z1), self.chiofz(z2)
        r1, r2 = chi1 / self.params.boxsize, chi2 / self.params.boxsize
        
        # Initialize empty data array with correct dtype (float32 to match file format)
        data = np.array([], dtype=np.float32).reshape(len(vars), 0)
        
        # Check redshift bounds
        if lightcone and z1 > self.params.zmax:
            return data
        elif not lightcone and step != np.argmin(np.abs(redshift - zvals)):
            return data
        
        # Process file chunks
        return self._process_chunks(step, bbox, vars, dtype, format_info, 
                                  lightcone, r1, r2, chunktask, data)

    def _process_chunks(self, step, bbox, vars, 
                       dtype, format_info, lightcone,
                       r1, r2, chunktask, data):
        """
        Process all chunks for a given step.
        
        For SLURM-level chunking (single file, this will always be the same file)
        and for local multiprocessing (for multiple files).
        """
        slurm_info = _get_slurm_info()
        
        # Determine if this is SLURM-level chunking (for single files)
        # vs local multiprocessing chunking (for multiple files)
        is_slurm_chunking = (slurm_info['is_slurm'] and 
                           0 <= chunktask < slurm_info['ntasks'])
        
        if is_slurm_chunking:
            # SLURM-level chunking: distribute chunks across SLURM processes
            ntasks_divisor = slurm_info['ntasks']
            task_id = chunktask  # chunktask is the SLURM_PROCID passed from _adaptive_tps_io
            _log_info(f"SLURM process {task_id}: Starting file read with {ntasks_divisor}-way chunking")
        else:
            # Local multiprocessing: distribute chunks across local processes
            ntasks_divisor = self.nproc
            task_id = chunktask
            if self.verbose:
                _log_info(f"Local process {task_id}: Starting file read with {ntasks_divisor}-way chunking")
        
        chunk = 0
        hoffset = format_info.get('hsize', 0)
        dsize = format_info.get('dsize', 1)
        count = -1
        
        # Configure chunking for tipsy format
        if format_info.get('name') == "tipsy":
            chunkmin = 2 * 1024**3
            count = chunkmin // dsize
            chunksize = count * dsize
            # Debug: Show chunk configuration (verbose mode only)
            if self.verbose and slurm_info['is_slurm']:
                _log_info(f"DEBUG: Chunk size: {chunksize/1024**3:.1f} GB, particles per chunk: {count:,}")
        else:
            # For non-tipsy formats (like lcp), read entire file in one chunk
            chunksize = 0
            _log_info(f"Non-tipsy format ({format_info.get('name', 'unknown')}): will read entire file")
        
        offset = hoffset
        
        # Use list to collect chunks, then concatenate once at the end (memory efficient)
        data_chunks = []
        
        # Production chunk processing limit for throughput optimization  
        chunks_processed = 0
        _log_info(f"🚀 CHUNK PROCESSING: Starting chunk processing (no limit - will process all chunks)")
        
        while True:
            # Debug: Log first few chunks to verify assignment (verbose mode only)
            if self.verbose and slurm_info['is_slurm'] and chunk < 3:
                _log_info(f"DEBUG: SLURM process {slurm_info['procid']} chunk {chunk}: assigned={(chunk % ntasks_divisor == task_id)}")
            
            # Skip chunks not assigned to this task
            if chunktask >= 0 and chunk % ntasks_divisor != task_id:
                chunk += 1
                offset += chunksize
                continue
            
            # Get current file (for single file, this will always be the same file)
            filepath = FileReader.get_filename(self.params.namespace, format_info, 0, step)
            chunk += 1
            
            # Read chunk data
            cdata = FileReader.read_chunk(filepath, dtype, offset, count)
            
            # Continue processing all assigned chunks
            assigned_chunks = sum(1 for i in range(chunk) if i % ntasks_divisor == task_id)
            
            if cdata.size == 0:
                if format_info.get('name') != "tipsy":
                    break
                # For tipsy format, concatenate all chunks and return
                if data_chunks:
                    return np.concatenate(data_chunks, axis=1)
                else:
                    return np.array([], dtype=np.float32).reshape(len(vars), 0)
            
            # For non-tipsy formats, process the data and break after first successful read
            if format_info.get('name') != "tipsy" and cdata.size > 0:
                # Process the data and break out - non-tipsy files are read entirely in one chunk
                _log_info(f"Non-tipsy format: read {cdata.size} elements, processing and exiting")
                # Continue with normal processing below, but will break at end of loop
            
            # Progress reporting for tipsy format
            if format_info.get('name') == "tipsy":
                ncum = (offset - 32 + chunksize) // dsize
                gbread = (offset - 32 + chunksize) / 1024**3
                
                # Progress reporting for tipsy format
                if is_slurm_chunking and _is_master_process():
                    try:
                        file_size = filepath.stat().st_size if filepath.exists() else 100 * 1024**3
                        progress_pct = min(100, (offset + chunksize) / file_size * 100)
                        print(f"SLURM process {task_id}: {progress_pct:.1f}% complete, {gbread:.1f} GB read, {ncum:,} particles", flush=True)
                    except Exception as e:
                        print(f"SLURM process {task_id}: {gbread:.1f} GB read, {ncum:,} particles", flush=True)
                elif not is_slurm_chunking and self.verbose:
                    print(f"Local task {task_id:>3d} read {ncum:>12} ({gbread:0.2f} GB)", flush=True)
                
                offset += chunksize
            
            # Process chunk and add to list (memory efficient)
            chunk_data = DataProcessor.cull_tile_reshape_single(cdata, vars, bbox, r1, r2, 
                                                              format_info, lightcone)
            if chunk_data.size > 0:
                # MEMORY OPTIMIZATION: Only strip velocity fields if we're doing position-only analysis
                # DO NOT strip velocity fields if they were explicitly requested (e.g., 'xvp' dataset)
                # This optimization should only apply when position data is sufficient for the analysis
                # For now, preserve all requested fields to maintain data integrity
                
                data_chunks.append(chunk_data)
            
            # Continue processing all chunks
            chunks_processed += 1
            
            # For non-tipsy formats, break after processing the first (and only) chunk
            if format_info.get('name') != "tipsy":
                _log_info(f"Non-tipsy format: completed processing, breaking out of chunk loop")
                break
        
        # Concatenate all chunks once at the end (memory efficient)
        if data_chunks:
            return np.concatenate(data_chunks, axis=1)
        else:
            return np.array([], dtype=np.float32).reshape(len(vars), 0)

    def _read_step_multiple_files(self, step, bbox, 
                                 dprops, format_info, 
                                 lightcone, redshift, task_id, 
                                 assigned_files):
        """Read data from multiple files assigned to this process.
        
        This method reads complete files (no within-file chunking) to avoid
        multiprocessing deadlocks that can occur with file locks.
        
        Args:
            step: Time step to read
            bbox: Bounding box for culling
            dprops: Dataset properties
            format_info: File format information
            lightcone: Whether to use lightcone mode
            redshift: Target redshift
            task_id: Process ID (for logging)
            assigned_files: List of file chunk numbers assigned to this process
            
        Returns:
            Combined data array from all assigned files
        """
        zvals = self.params.zoutput
        
        # Validate step index
        if step >= len(zvals) or step < 1:
            logger.warning(f"Invalid step {step}, must be between 1 and {len(zvals)-1}")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        z1, z2 = zvals[step], zvals[step - 1]
        vars = dprops['vars']
        dtype = format_info['dtype']
        
        # Calculate distance bounds
        chi1, chi2 = self.chiofz(z1), self.chiofz(z2)
        r1, r2 = chi1 / self.params.boxsize, chi2 / self.params.boxsize
        
        # Initialize empty data array
        data = np.array([]).reshape(len(vars), 0)
        
        # Check redshift bounds
        if lightcone and z1 > self.params.zmax:
            return data
        elif not lightcone and step != np.argmin(np.abs(redshift - zvals)):
            return data
        
        # Process assigned files (complete files, not chunks within files)
        files_processed = 0
        total_size_gb = 0
        
        for file_chunk in assigned_files:
            # Get file path
            filepath = FileReader.get_filename(self.params.namespace, format_info, file_chunk, step)
            
            if not filepath.exists():
                continue
            
            # Read entire file (no offset, no count limit)
            file_size = filepath.stat().st_size
            total_size_gb += file_size / (1024**3)
            
            cdata = FileReader.read_chunk(filepath, dtype, offset=0, count=-1)
            
            if cdata.size == 0:
                continue
            
            # Process and add data
            data = DataProcessor.cull_tile_reshape(data, cdata, vars, bbox, r1, r2, 
                                                 format_info, lightcone)
            files_processed += 1
            
            # Progress reporting
            if self.verbose and _is_master_process():
                print(f"Task {task_id:>3d}: processed file {file_chunk} "
                      f"({files_processed}/{len(assigned_files)} files, "
                      f"{total_size_gb:.2f} GB total)", end='\r', flush=True)
        
        if self.verbose and _is_master_process() and files_processed > 0:
            print(f"\nTask {task_id:>3d}: completed {files_processed} files, "
                  f"{data.shape[1]:,} particles, {total_size_gb:.2f} GB")
        
        return data

    def _adaptive_tps_io(self, step, bbox, dprops, format_info, redshift):
        """
        Adaptive TPS file I/O using multi-level parallel strategy.
        
        ENHANCED VERSION: Implements high-throughput multiprocessing I/O as Phase 1.
        
        Strategy based on file count:
        1. Single file: SLURM-level chunking with multiprocessing I/O within each process
        2. Few files (≤ SLURM_NTASKS): One file per SLURM process + multiprocessing chunks  
        3. Many files: Distribute across SLURM processes + multiprocessing within each
        
        Args:
            step: Time step to read
            bbox: Bounding box for culling
            dprops: Dataset properties
            format_info: File format information  
            redshift: Target redshift
            
        Returns:
            Combined data array from all assigned files/chunks
        """
        slurm_info = _get_slurm_info()
        
        # Detect available chunk files for this step
        max_search = max(slurm_info['ntasks'] * slurm_info['cpus_per_task'], 100)
        available_files = []
        chunk = 0
        while chunk < max_search:
            try:
                filepath = FileReader.get_filename(self.params.namespace, format_info, chunk, step)
                if filepath.exists():
                    available_files.append(chunk)
                    chunk += 1
                else:
                    break
            except:
                break
        
        n_files = len(available_files)
        ntasks = slurm_info['ntasks']
        procid = slurm_info['procid']
        cpus_per_task = slurm_info['cpus_per_task']
        
        if n_files == 0:
            _log_info("No TPS files found")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        elif n_files == 1:
            # SINGLE FILE: Revert to proven sequential I/O (avoid OOM)
            _log_info(f"🚀 MEMORY-SAFE I/O: Single file using proven sequential chunking (avoiding OOM)")
            _log_info(f"   File size: {FileReader.get_filename(self.params.namespace, format_info, 0, step).stat().st_size / 1024**3:.2f} GB")
            _log_info(f"   Available cores: {cpus_per_task} per SLURM process, but using sequential I/O for memory safety")
            
            # Use the proven sequential approach
            start_time = time.time()
            result = self._read_step(step, bbox, dprops, format_info, False, redshift, procid)
            end_time = time.time()
            
            io_time = end_time - start_time
            particles = result.shape[1]
            
            _log_info(f"🚀 MEMORY-SAFE I/O COMPLETE: SLURM process {procid}: {particles:,} particles in {io_time:.1f}s")
            
            return result
        
        elif n_files <= ntasks:
            # FEW FILES: One complete file per SLURM process + potential for enhanced chunked reading
            assigned_files = [f for i, f in enumerate(available_files) if i % ntasks == procid]
            
            if not assigned_files:
                _log_info(f"No files assigned to SLURM process {procid}")
                return np.array([]).reshape(len(dprops['vars']), 0)
            
            _log_info(f"🚀 ENHANCED I/O: Few files ({n_files}) - SLURM process {procid} reading {len(assigned_files)} files")
            
            # For now, read assigned files sequentially but optimize for future multiprocessing
            all_data = []
            for file_chunk in assigned_files:
                file_data = self._read_single_file_complete(step, bbox, dprops, format_info, 
                                                          False, redshift, file_chunk)
                if file_data.size > 0:
                    all_data.append(file_data)
            
            return np.concatenate(all_data, axis=1) if all_data else np.array([]).reshape(len(dprops['vars']), 0)
        
        else:
            # MANY FILES: Distribute across SLURM processes + local multiprocessing
            assigned_files = [f for i, f in enumerate(available_files) if i % ntasks == procid]
            
            if not assigned_files:
                _log_info(f"No files assigned to SLURM process {procid}")
                return np.array([]).reshape(len(dprops['vars']), 0)
            
            effective_nproc = min(len(assigned_files), cpus_per_task)
            _log_info(f"🚀 ENHANCED I/O: Many files ({n_files}) - SLURM process {procid} using {effective_nproc} cores for {len(assigned_files)} files")
            
            if effective_nproc > 1:
                # Use multiprocessing within this SLURM process
                args = []
                for task in range(effective_nproc):
                    task_files = [f for i, f in enumerate(assigned_files) if i % effective_nproc == task]
                    if task_files:
                        args.append([step, bbox, dprops, format_info, False, redshift, task, task_files])
                
                with mp.Pool(processes=effective_nproc) as pool:
                    results = pool.starmap(self._read_step_multiple_files, args)
                return np.concatenate([r for r in results if r.size > 0], axis=1)
            else:
                # Single process handles all assigned files
                return self._read_step_multiple_files(step, bbox, dprops, format_info, 
                                                    False, redshift, 0, assigned_files)

    def _read_single_file_complete(self, step, bbox, dprops, format_info, 
                                  lightcone, redshift, file_chunk):
        """
        Read a complete single file (for few-files scenario).
        
        Args:
            step: Time step to read
            bbox: Bounding box for culling
            dprops: Dataset properties
            format_info: File format information
            lightcone: Whether in lightcone mode
            redshift: Target redshift
            file_chunk: Which file chunk to read
            
        Returns:
            Data array from the complete file
        """
        zvals = self.params.zoutput
        
        if step >= len(zvals) or step < 1:
            logger.warning(f"Invalid step {step}")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        z1, z2 = zvals[step], zvals[step - 1]
        vars = dprops['vars']
        dtype = format_info['dtype']
        
        # Calculate distance bounds
        chi1, chi2 = self.chiofz(z1), self.chiofz(z2)
        r1, r2 = chi1 / self.params.boxsize, chi2 / self.params.boxsize
        
        # Check redshift bounds
        if lightcone and z1 > self.params.zmax:
            return np.array([]).reshape(len(vars), 0)
        elif not lightcone and step != np.argmin(np.abs(redshift - zvals)):
            return np.array([]).reshape(len(vars), 0)
        
        # Read the complete file
        filepath = FileReader.get_filename(self.params.namespace, format_info, file_chunk, step)
        
        if not filepath.exists():
            return np.array([]).reshape(len(vars), 0)
        
        # Read entire file
        cdata = FileReader.read_chunk(filepath, dtype, offset=0, count=-1)
        
        if cdata.size == 0:
            return np.array([]).reshape(len(vars), 0)
        
        # Process data
        data = np.array([]).reshape(len(vars), 0)
        return DataProcessor.cull_tile_reshape(data, cdata, vars, bbox, r1, r2, 
                                             format_info, lightcone)

    def fetch_data(self, bbox, dataset = 'xvp', 
                   filetype = 'lcp', lightcone = False, 
                   redshifts = [0]):
        """Fetch particle data from simulation files.
        
        Args:
            bbox: Bounding box [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
            dataset: Dataset type ('xvp', etc.)
            filetype: File format ('lcp', 'tps', 'fof')
            lightcone: Whether to use lightcone mode
            redshifts: List of redshifts to fetch (for non-lightcone mode)
            
        Returns:
            Data array/dictionary
            
        Raises:
            ValueError: If dataset or filetype is invalid
        """
        # Validate inputs
        if dataset not in pkds.properties:
            raise ValueError(f"Unknown dataset: {dataset}")
        if filetype not in pkds.fileformats:
            raise ValueError(f"Unknown filetype: {filetype}")
        
        dprops = pkds.properties[dataset]
        format_info = pkds.fileformats[filetype]
        vars = dprops['vars']
        
        _log_info(f"Fetching {dataset} data with {filetype} format")
        _log_info(f"Lightcone: {lightcone}, Variables: {vars}")
        _log_info(f"Bounding box: {bbox}")
        
        # Load all data into memory
        if lightcone:
            return self._fetch_lightcone_data(bbox, dprops, format_info, vars)
        else:
            return self._fetch_snapshot_data(bbox, dprops, format_info, vars, 
                                           filetype, redshifts)

    def _fetch_lightcone_data(self, bbox, dprops, 
                             format_info, vars):
        """Fetch data in lightcone mode."""
        max_step = len(self.params.zoutput) - 1
        args = [[step, bbox, dprops, format_info, True, 0, -1] 
                for step in range(1, max_step + 1)]
        
        # Use serial execution for small datasets or test environments to avoid hanging
        if max_step <= 3 or self.nproc == 1:
            _log_info(f"Using serial execution for lightcone data (max_step={max_step}, nproc={self.nproc})")
            results = [self._read_step(*arg) for arg in args]
        else:
            _log_info(f"Using parallel execution for lightcone data (max_step={max_step}, nproc={self.nproc})")
            with mp.Pool(processes=self.nproc) as pool:
                results = pool.starmap(self._read_step, args)
        
        # Handle empty results
        if not results or all(len(r) == 0 for r in results):
            _log_info("No lightcone data found in specified bbox")
            return np.array([], dtype=[]).reshape(0,)
        
        # Filter out empty results before concatenation
        non_empty_results = [r for r in results if len(r) > 0]
        if not non_empty_results:
            _log_info("All lightcone results were empty")
            return np.array([], dtype=[]).reshape(0,)
        
        concatenated = np.concatenate(non_empty_results, axis=1)
        # Wrap in array to maintain backward compatibility with legacy format (1, N) shape
        data = np.asarray([np.rec.fromarrays(concatenated, names=vars)])
        
        _log_info(f"Lightcone data shape: {data.shape}")
        return data

    def _fetch_snapshot_data(self, bbox, dprops, 
                            format_info, vars, 
                            filetype, redshifts):
        """Fetch data in snapshot mode."""
        data = {}
        zout = []
        
        for i, redshift in enumerate(redshifts):
            step = np.argmin(np.abs(redshift - self.params.zoutput))
            zout.append(self.params.zoutput[step])
            
            if filetype == 'fof':
                array = self._read_step(step, bbox, dprops, format_info, 
                                      False, redshift, -1)
            elif filetype == 'tps':
                # Implement adaptive multi-level parallel I/O strategy
                array = self._adaptive_tps_io(step, bbox, dprops, format_info, redshift)
            else:
                raise ValueError(f"Unsupported filetype for snapshots: {filetype}")
            
            # BUGFIX: Account for memory optimization that may strip velocity fields
            # The array may have fewer fields than originally requested
            actual_nfields = array.shape[0] if array.size > 0 else len(vars)
            
            if array.size > 0 and array.size % actual_nfields != 0:
                # This should never happen with our corrected logic
                raise ValueError(f"Array size {array.size} not divisible by {actual_nfields} fields")
            
            # Use only the variables that match the actual array structure
            if actual_nfields == 3 and len(vars) == 6:
                # Memory optimization stripped velocity fields, use only position vars
                actual_vars = vars[:3]  # ['x', 'y', 'z']
                _log_info(f"Memory optimization detected: using {actual_vars} (velocities stripped)")
            else:
                actual_vars = vars
            
            data[f"box{i}"] = np.rec.fromarrays(array.reshape(actual_nfields, -1), names=actual_vars)
        
        self.zout = np.array(zout)
        self.nout = len(data)
        
        _log_info(f"Snapshot data: {len(data)} boxes")
        for key, value in data.items():
            _log_info(f"  {key}: shape {value.shape}")
        
        return data

    def matter_power(self, bbox, ngrid = 256, n_devices = 1, 
                    subtract_shot_noise = False, **kwargs):
        """
        Compute matter power spectrum for given bounding box.
        
        Args:
            bbox: Bounding box [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
            ngrid: Grid resolution for FFT
            n_devices: Number of GPUs to use
            subtract_shot_noise: Whether to subtract shot noise
            **kwargs: Additional arguments passed to fetch_data
            
        Returns:
            Tuple of (k_bins, power_spectrum, n_modes_per_bin)
        """
        from .power_spectrum import PowerSpectrumCalculator
        
        # Fetch particle data
        particle_data = self.fetch_data(bbox, **kwargs)
        
        # Handle different return formats from fetch_data
        if isinstance(particle_data, dict):
            # Snapshot mode - use first box
            particles = list(particle_data.values())[0]
        else:
            # Lightcone mode
            particles = particle_data[0] if len(particle_data) > 0 else particle_data
        
        # Convert to format expected by PowerSpectrumCalculator
        particle_dict = {
            'x': particles['x'],
            'y': particles['y'], 
            'z': particles['z'],
            'mass': np.ones(len(particles))  # Assume equal mass
        }
        
        # Calculate box size from bbox
        box_size = bbox[0][1] - bbox[0][0]  # Assume cubic box
        
        # Calculate power spectrum
        calculator = PowerSpectrumCalculator(
            ngrid=ngrid,
            box_size=box_size,
            n_devices=n_devices
        )
        
        return calculator.calculate_power_spectrum(
            particle_dict,
            subtract_shot_noise=subtract_shot_noise
        )

    def get_simulation_parameters(self):
        """
        Get simulation parameters as a dictionary.
        
        Returns:
            Dictionary containing simulation parameters
        """
        return {
            'namespace': self.params.namespace,
            'zmax': self.params.zmax,
            'dBoxSize': self.params.boxsize,  # Use dBoxSize for compatibility
            'box_size': self.params.boxsize,   # Also provide box_size alias
            'nsteps': self.params.nsteps,
            'omegam': self.params.omegam,
            'h': self.params.h,
            'ngrid': self.params.ngrid,
            'zoutput': self.params.zoutput
        }
        """
        Get simulation parameters as a dictionary.
        
        Returns:
            Dictionary containing simulation parameters
        """
        return {
            'namespace': self.params.namespace,
            'zmax': self.params.zmax,
            'dBoxSize': self.params.boxsize,  # Use dBoxSize for compatibility
            'box_size': self.params.boxsize,   # Also provide box_size alias
            'nsteps': self.params.nsteps,
            'omegam': self.params.omegam,
            'h': self.params.h,
            'ngrid': self.params.ngrid,
            'zoutput': self.params.zoutput
        }

    def _parallel_chunked_io_single_file(self, step, bbox, dprops, format_info, redshift, slurm_info):
        """
        Enhanced I/O for single file: Each SLURM process spawns multiple workers for chunk reading.
        
        This implements Phase 1 of the multiprocessing I/O plan:
        - Each SLURM process gets chunks assigned via SLURM_PROCID
        - Within each SLURM process, spawn workers to read chunks in parallel
        - Workers process their assigned chunks and return particles
        - SLURM process consolidates all worker results
        
        Args:
            step: Time step to read
            bbox: Bounding box for culling  
            dprops: Dataset properties
            format_info: File format information
            redshift: Target redshift
            slurm_info: SLURM environment information
            
        Returns:
            Combined data array from all chunks assigned to this SLURM process
        """
        procid = slurm_info['procid']
        ntasks = slurm_info['ntasks']
        cpus_per_task = slurm_info['cpus_per_task']
        
        # Determine chunks for this SLURM process using existing logic
        # Get file path for single file (chunk 0)
        filepath = FileReader.get_filename(self.params.namespace, format_info, 0, step)
        
        if not filepath.exists():
            _log_info(f"SLURM process {procid}: File not found: {filepath}")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        # Calculate file size and chunking parameters
        file_size = filepath.stat().st_size
        hoffset = format_info.get('hsize', 0)
        dsize = format_info.get('dsize', 1)
        
        # Configure chunking for tipsy format (same as existing logic)
        if format_info.get('name') == "tipsy":
            chunkmin = 2 * 1024**3  # 2GB chunks
            particles_per_chunk = chunkmin // dsize
            chunk_size_bytes = particles_per_chunk * dsize
        else:
            # For other formats, use smaller chunks
            particles_per_chunk = 1024**2  # 1M particles per chunk
            chunk_size_bytes = particles_per_chunk * dsize
        
        # Calculate total number of chunks in file
        data_size = file_size - hoffset
        total_chunks = max(1, (data_size + chunk_size_bytes - 1) // chunk_size_bytes)
        
        # Determine which chunks belong to this SLURM process
        my_chunks = [c for c in range(total_chunks) if c % ntasks == procid]
        
        if not my_chunks:
            _log_info(f"SLURM process {procid}: No chunks assigned")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        _log_info(f"🚀 PARALLEL I/O: SLURM process {procid} processing {len(my_chunks)} chunks with {cpus_per_task} workers")
        _log_info(f"🚀 PARALLEL I/O: File size: {file_size/1024**3:.2f} GB, chunk size: {chunk_size_bytes/1024**3:.2f} GB")
        
        # Distribute chunks among workers within this SLURM process
        effective_workers = min(len(my_chunks), cpus_per_task)
        
        if effective_workers == 1:
            # Single worker handles all chunks for this SLURM process
            return self._read_chunks_sequentially(filepath, my_chunks, step, bbox, dprops, 
                                                format_info, redshift, hoffset, 
                                                particles_per_chunk, procid)
        
        # Multiple workers: distribute chunks among them
        worker_args = []
        for worker_id in range(effective_workers):
            worker_chunks = [c for i, c in enumerate(my_chunks) if i % effective_workers == worker_id]
            if worker_chunks:
                # Enhanced arguments with all necessary data for workers
                enhanced_dprops = dict(dprops)
                enhanced_dprops['namespace'] = self.params.namespace
                enhanced_dprops['zvals'] = self.params.zoutput
                
                worker_args.append([
                    str(filepath), worker_chunks, step, bbox, enhanced_dprops, 
                    format_info, redshift, hoffset, particles_per_chunk, 
                    procid, worker_id
                ])
        
        # Launch I/O workers 
        start_time = time.time()
        with mp.Pool(processes=effective_workers) as pool:
            try:
                worker_results = pool.starmap(self._io_worker_process_chunks, worker_args)
            except Exception as e:
                _log_info(f"🚨 PARALLEL I/O ERROR: SLURM process {procid}: Worker pool failed: {e}")
                # Fallback to sequential processing
                return self._read_chunks_sequentially(filepath, my_chunks, step, bbox, dprops,
                                                    format_info, redshift, hoffset,
                                                    particles_per_chunk, procid)
        
        end_time = time.time()
        
        # Consolidate results from all workers
        valid_results = [r for r in worker_results if r.size > 0]
        
        if valid_results:
            combined_data = np.concatenate(valid_results, axis=1)
            total_particles = combined_data.shape[1]
            throughput_gb_s = (len(my_chunks) * chunk_size_bytes / 1024**3) / (end_time - start_time)
            
            _log_info(f"🚀 PARALLEL I/O SUCCESS: SLURM process {procid}: {total_particles:,} particles, "
                     f"{throughput_gb_s:.1f} GB/s, {end_time - start_time:.1f}s")
            
            return combined_data
        else:
            _log_info(f"SLURM process {procid}: No valid data from workers")
            return np.array([]).reshape(len(dprops['vars']), 0)

    @staticmethod
    def _io_worker_process_chunks(filepath_str, chunk_list, step, bbox, dprops, 
                                 format_info, redshift, hoffset, particles_per_chunk, 
                                 slurm_procid, worker_id):
        """
        Worker process for reading assigned chunks from a single file.
        
        This is a static method to ensure it can be pickled for multiprocessing.
        Each worker reads its assigned chunks sequentially and returns combined data.
        
        Args:
            filepath_str: File path as string (for pickling)
            chunk_list: List of chunk numbers assigned to this worker
            step: Time step
            bbox: Bounding box for culling
            dprops: Dataset properties
            format_info: File format information
            redshift: Target redshift  
            hoffset: Header offset in bytes
            particles_per_chunk: Number of particles per chunk
            slurm_procid: SLURM process ID (for logging)
            worker_id: Worker ID within SLURM process
            
        Returns:
            Combined data array from all assigned chunks
        """
        try:
            filepath = Path(filepath_str)
            vars = dprops['vars']
            dtype = format_info['dtype']
            dsize = format_info.get('dsize', 1)
            chunk_size_bytes = particles_per_chunk * dsize
            
            # Load simulation parameters (needed for DataProcessor)
            # Since this is a static method, we need to reconstruct the cosmological interpolation
            zvals = dprops.get('zvals')  # We'll need to pass this in the args
            if zvals is None:
                # Fallback: just process without redshift filtering
                z1, z2, r1, r2 = 0, 0, 0, 1
            else:
                if step >= len(zvals) or step < 1:
                    return np.array([]).reshape(len(vars), 0)
                z1, z2 = zvals[step], zvals[step - 1]
                # Simple distance calculation (without full cosmology)
                r1, r2 = z1 / 1000, z2 / 1000  # Simplified for worker
            
            data_chunks = []
            
            for chunk_num in chunk_list:
                # Calculate byte offset for this chunk
                offset = hoffset + chunk_num * chunk_size_bytes
                
                try:
                    # Read chunk data
                    cdata = np.fromfile(filepath, dtype=dtype, offset=offset, count=particles_per_chunk)
                    
                    if cdata.size == 0:
                        continue
                    
                    # Process chunk using simplified culling (avoid full DataProcessor to prevent import issues)
                    chunk_data = Data._simplified_cull_reshape(cdata, vars, bbox)
                    
                    if chunk_data.size > 0:
                        data_chunks.append(chunk_data)
                        
                except (IOError, ValueError) as e:
                    # Log error but continue with other chunks
                    print(f"Worker {worker_id}: Failed to read chunk {chunk_num}: {e}", flush=True)
                    continue
            
            # Combine all chunks from this worker
            if data_chunks:
                combined = np.concatenate(data_chunks, axis=1)
                print(f"Worker {worker_id}: Processed {len(chunk_list)} chunks → {combined.shape[1]:,} particles", flush=True)
                return combined
            else:
                return np.array([]).reshape(len(vars), 0)
                
        except Exception as e:
            print(f"🚨 Worker {worker_id} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return np.array([]).reshape(len(dprops['vars']), 0)

    def _enhanced_single_file_io(self, step, bbox, dprops, format_info, redshift, slurm_info):
        """
        Enhanced single file I/O using multiprocessing for chunk reading.
        
        This implements the core multiprocessing I/O enhancement:
        - Determine chunks assigned to this SLURM process
        - Distribute those chunks among available workers
        - Workers read chunks in parallel and return particle data
        - Consolidate results from all workers
        
        Args:
            step: Time step to read
            bbox: Bounding box for culling
            dprops: Dataset properties
            format_info: File format information
            redshift: Target redshift
            slurm_info: SLURM environment information
            
        Returns:
            Combined data array from all chunks assigned to this SLURM process
        """
        procid = slurm_info['procid']
        ntasks = slurm_info['ntasks']
        cpus_per_task = slurm_info['cpus_per_task']
        
        # Get file information
        filepath = FileReader.get_filename(self.params.namespace, format_info, 0, step)
        
        if not filepath.exists():
            _log_info(f"SLURM process {procid}: File not found: {filepath}")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        # Enhanced dynamic chunking strategy for maximum core utilization
        file_size = filepath.stat().st_size
        hoffset = format_info.get('hsize', 0)
        dsize = format_info.get('dsize', 1)
        data_size = file_size - hoffset
        
        # Calculate optimal chunk size to utilize all available cores
        total_cores = ntasks * cpus_per_task  # e.g., 4 × 32 = 128 cores
        max_chunk_size = 1 * 1024**3  # 1GB max chunk size
        
        # Determine chunk size: smaller of max_chunk_size or file_size / total_cores
        optimal_chunk_size = min(max_chunk_size, max(64 * 1024**2, data_size // total_cores))  # min 64MB chunks
        particles_per_chunk = optimal_chunk_size // dsize
        chunk_size_bytes = particles_per_chunk * dsize
        
        # Calculate total number of chunks needed
        total_chunks = max(1, (data_size + chunk_size_bytes - 1) // chunk_size_bytes)
        
        _log_info(f"🚀 DYNAMIC CHUNKING: File size: {file_size/1024**3:.2f} GB, "
                 f"Target: {total_cores} cores, Chunk size: {chunk_size_bytes/1024**3:.2f} GB, "
                 f"Total chunks: {total_chunks}")
        
        # Calculate which chunks belong to this SLURM process
        my_chunks = [c for c in range(total_chunks) if c % ntasks == procid]
        
        if not my_chunks:
            _log_info(f"SLURM process {procid}: No chunks assigned")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        _log_info(f"🚀 ENHANCED I/O: SLURM process {procid} processing {len(my_chunks)} chunks with up to {cpus_per_task} workers")
        
        # Use all available cores if we have enough chunks, otherwise use sequential processing
        if len(my_chunks) == 1:
            _log_info(f"🚀 ENHANCED I/O: Single chunk - using sequential processing")
            return self._process_chunks_sequentially(my_chunks, filepath, step, bbox, dprops, 
                                                   format_info, redshift, hoffset, 
                                                   particles_per_chunk, procid)
        
        # Use multiprocessing: maximize worker utilization
        effective_workers = min(len(my_chunks), cpus_per_task)
        _log_info(f"🚀 ENHANCED I/O: Using {effective_workers} workers for {len(my_chunks)} chunks (worker efficiency: {len(my_chunks)/effective_workers:.1f} chunks/worker)")
        
        # Prepare worker arguments
        worker_args = []
        for worker_id in range(effective_workers):
            worker_chunks = [c for i, c in enumerate(my_chunks) if i % effective_workers == worker_id]
            if worker_chunks:
                worker_args.append([
                    str(filepath), worker_chunks, step, bbox, dprops, format_info, 
                    redshift, hoffset, particles_per_chunk, chunk_size_bytes, procid, worker_id
                ])
        
        # Launch worker pool
        start_time = time.time()
        _log_info(f"🚀 MEMORY-OPTIMIZED I/O: Launching {len(worker_args)} workers for {len(my_chunks)} chunks")
        _log_info(f"🚀 MEMORY PROFILE: Using optimized worker with batch processing and immediate cleanup")
        
        try:
            with mp.Pool(processes=effective_workers) as pool:
                worker_results = pool.starmap(self._multiproc_chunk_reader_worker, worker_args)
        except Exception as e:
            _log_info(f"🚨 MEMORY-OPTIMIZED I/O ERROR: Worker pool failed: {e}")
            # Fallback to sequential processing
            return self._process_chunks_sequentially(my_chunks, filepath, step, bbox, dprops,
                                                   format_info, redshift, hoffset,
                                                   particles_per_chunk, procid)
        
        end_time = time.time()
        
        # Consolidate results
        valid_results = [r for r in worker_results if r.size > 0]
        
        if valid_results:
            combined_data = np.concatenate(valid_results, axis=1)
            total_particles = combined_data.shape[1]
            io_time = end_time - start_time
            data_processed_gb = (len(my_chunks) * chunk_size_bytes / 1024**3)
            throughput_gb_s = data_processed_gb / io_time
            core_efficiency = len(my_chunks) / effective_workers
            
            _log_info(f"🚀 ENHANCED I/O SUCCESS: SLURM process {procid}: {total_particles:,} particles, "
                     f"{throughput_gb_s:.1f} GB/s, {io_time:.1f}s with {effective_workers} workers "
                     f"({core_efficiency:.1f} chunks/worker, {data_processed_gb:.1f} GB total)")
            
            return combined_data
        else:
            _log_info(f"SLURM process {procid}: No valid data from workers")
            return np.array([]).reshape(len(dprops['vars']), 0)

    @staticmethod  
    def _multiproc_chunk_reader_worker(filepath_str, chunk_list, step, bbox, dprops, format_info,
                                      redshift, hoffset, particles_per_chunk, chunk_size_bytes, 
                                      slurm_procid, worker_id):
        """
        Worker function for reading assigned chunks in parallel.
        
        This is a static method to ensure it can be pickled for multiprocessing.
        Each worker reads its assigned chunks and returns particle data.
        
        Args:
            filepath_str: Path to file (as string for pickling)
            chunk_list: List of chunk numbers assigned to this worker
            step: Time step
            bbox: Bounding box for culling
            dprops: Dataset properties
            format_info: File format information
            redshift: Target redshift
            hoffset: Header offset in bytes
            particles_per_chunk: Number of particles per chunk
            chunk_size_bytes: Size of each chunk in bytes
            slurm_procid: SLURM process ID (for logging)
            worker_id: Worker ID within SLURM process
            
        Returns:
            Combined data array from all assigned chunks
        """
        try:
            filepath = Path(filepath_str)
            vars = dprops['vars']
            dtype = format_info['dtype']
            
            if not filepath.exists():
                return np.array([]).reshape(len(vars), 0)
            
            data_chunks = []
            bytes_read = 0
            max_chunks_in_memory = 4  # Process chunks in batches to reduce memory
            
            for i, chunk_num in enumerate(chunk_list):
                try:
                    # Calculate byte offset for this chunk
                    offset = hoffset + chunk_num * chunk_size_bytes
                    
                    # Read chunk data
                    cdata = np.fromfile(filepath, dtype=dtype, offset=offset, count=particles_per_chunk)
                    
                    if cdata.size == 0:
                        continue
                    
                    bytes_read += cdata.nbytes
                    
                    # Apply coordinate shifts and culling in one pass (memory efficient)
                    format_offset = format_info.get('offset', [0, 0, 0])
                    coord_shift = -np.asarray(format_offset)  # Negate offset as in DataProcessor
                    
                    # Apply spatial culling with coordinate shifts (avoid intermediate arrays)
                    if 'x' in cdata.dtype.names and 'y' in cdata.dtype.names and 'z' in cdata.dtype.names:
                        # Apply bounding box directly on shifted coordinates (no intermediate arrays)
                        xmin, xmax = bbox[0][0], bbox[0][1]
                        ymin, ymax = bbox[1][0], bbox[1][1]
                        zmin, zmax = bbox[2][0], bbox[2][1]
                        
                        # Create mask using shifted coordinates directly (no copies)
                        mask = (((cdata['x'] + coord_shift[0]) >= xmin) & ((cdata['x'] + coord_shift[0]) <= xmax) & 
                               ((cdata['y'] + coord_shift[1]) >= ymin) & ((cdata['y'] + coord_shift[1]) <= ymax) & 
                               ((cdata['z'] + coord_shift[2]) >= zmin) & ((cdata['z'] + coord_shift[2]) <= zmax))
                        
                        if np.any(mask):
                            # Extract and process only the variables we need (single pass)
                            extracted_arrays = []
                            for var in vars:
                                if var in cdata.dtype.names:
                                    # Apply mask and coordinate shift in one operation
                                    if var == 'x':
                                        data_array = (cdata[var][mask] + coord_shift[0]).astype(np.float32)
                                    elif var == 'y':
                                        data_array = (cdata[var][mask] + coord_shift[1]).astype(np.float32)
                                    elif var == 'z':
                                        data_array = (cdata[var][mask] + coord_shift[2]).astype(np.float32)
                                    else:
                                        data_array = cdata[var][mask].astype(np.float32)
                                    extracted_arrays.append(data_array)
                                else:
                                    extracted_arrays.append(np.zeros(np.sum(mask), dtype=np.float32))
                            
                            if extracted_arrays and len(extracted_arrays[0]) > 0:
                                chunk_data = np.stack(extracted_arrays, axis=0)
                                data_chunks.append(chunk_data)
                            
                            # Force cleanup of intermediate arrays
                            del extracted_arrays
                    
                    # Force cleanup of chunk data immediately
                    del cdata
                    
                    # Process chunks in batches to reduce memory usage
                    if len(data_chunks) >= max_chunks_in_memory and i < len(chunk_list) - 1:
                        # Consolidate current batch and continue
                        if len(data_chunks) > 1:
                            batch_combined = np.concatenate(data_chunks, axis=1)
                            data_chunks = [batch_combined]  # Replace multiple chunks with one combined
                        # Force garbage collection
                        import gc
                        gc.collect()
                    
                except (IOError, ValueError) as e:
                    print(f"Worker {worker_id}: Failed to read chunk {chunk_num}: {e}", flush=True)
                    continue
            
            # Combine all chunks from this worker
            if data_chunks:
                combined = np.concatenate(data_chunks, axis=1)
                gb_read = bytes_read / 1024**3
                print(f"Worker {worker_id}: Processed {len(chunk_list)} chunks → {combined.shape[1]:,} particles, {gb_read:.2f} GB", flush=True)
                return combined
            else:
                return np.array([]).reshape(len(vars), 0)
                
        except Exception as e:
            print(f"🚨 Worker {worker_id} FAILED: {e}", flush=True)
            return np.array([]).reshape(len(dprops['vars']), 0)

    def _process_chunks_sequentially(self, chunk_list, filepath, step, bbox, dprops,
                                   format_info, redshift, hoffset, particles_per_chunk, procid):
        """
        Fallback sequential chunk processing.
        
        Args:
            chunk_list: List of chunk numbers to process
            filepath: Path to file
            step: Time step
            bbox: Bounding box
            dprops: Dataset properties
            format_info: File format information
            redshift: Target redshift
            hoffset: Header offset
            particles_per_chunk: Particles per chunk
            procid: Process ID for logging
            
        Returns:
            Combined data from all chunks
        """
        _log_info(f"SLURM process {procid}: Processing {len(chunk_list)} chunks sequentially")
        
        vars = dprops['vars']
        dtype = format_info['dtype']
        dsize = format_info.get('dsize', 1)
        chunk_size_bytes = particles_per_chunk * dsize
        
        data_chunks = []
        
        for chunk_num in chunk_list:
            try:
                offset = hoffset + chunk_num * chunk_size_bytes
                cdata = FileReader.read_chunk(filepath, dtype, offset, particles_per_chunk)
                
                if cdata.size == 0:
                    continue
                
                # Use existing data processing
                chunk_data = DataProcessor.cull_tile_reshape_single(cdata, vars, bbox, 0, 1, 
                                                                  format_info, False)
                if chunk_data.size > 0:
                    data_chunks.append(chunk_data)
                    
            except Exception as e:
                _log_info(f"Failed to read chunk {chunk_num}: {e}")
                continue
        
        if data_chunks:
            return np.concatenate(data_chunks, axis=1)
        else:
            return np.array([]).reshape(len(vars), 0)
    
    def _read_files_sequentially(self, step, bbox, dprops, format_info, 
                                redshift, file_list, procid):
        """
        Fallback sequential file reading when multiprocessing fails.
        
        Args:
            step: Time step
            bbox: Bounding box  
            dprops: Dataset properties
            format_info: File format information
            redshift: Target redshift
            file_list: List of file chunk numbers to read
            procid: Process ID for logging
            
        Returns:
            Combined data from all files
        """
        _log_info(f"SLURM process {procid}: Falling back to sequential file reading")
        
        data_chunks = []
        
        for file_chunk in file_list:
            try:
                file_data = self._read_single_file_complete(step, bbox, dprops, format_info,
                                                          False, redshift, file_chunk)
                if file_data.size > 0:
                    data_chunks.append(file_data)
            except Exception as e:
                _log_info(f"Failed to read file {file_chunk}: {e}")
                continue
        
        if data_chunks:
            return np.concatenate(data_chunks, axis=1)
        else:
            return np.array([]).reshape(len(dprops['vars']), 0)
