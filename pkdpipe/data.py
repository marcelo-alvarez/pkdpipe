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
        
        # Apply shifts
        for var in ['x', 'y', 'z']:
            if var in cdata.dtype.names:
                cdata[var] += shift[var]
        
        # Apply spatial bounds
        if all(var in vars for var in ['x', 'y', 'z']):
            mask = ((cdata['x'] > bounds[0][0]) & (cdata['x'] <= bounds[0][1]) &
                   (cdata['y'] > bounds[1][0]) & (cdata['y'] <= bounds[1][1]) &
                   (cdata['z'] > bounds[2][0]) & (cdata['z'] <= bounds[2][1]))
            
            # Apply radial bounds if specified
            if len(bounds) > 3:
                r = np.sqrt(cdata['x']**2 + cdata['y']**2 + cdata['z']**2)
                mask = mask & (r > bounds[3][0]) & (r <= bounds[3][1])
        else:
            mask = np.ones(len(cdata), dtype=bool)
        
        # Extract and reshape selected data
        try:
            result = np.concatenate([cdata[var][mask] for var in vars])
            return result.reshape((len(vars), -1))
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
            data = np.concatenate((data, sdata), axis=1)
        
        return data


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
        
        # Process file chunks
        return self._process_chunks(step, bbox, vars, dtype, format_info, 
                                  lightcone, r1, r2, chunktask, data)

    def _process_chunks(self, step, bbox, vars, 
                       dtype, format_info, lightcone,
                       r1, r2, chunktask, data):
        """
        Process all chunks for a given step.
        
        For SLURM-level chunking (single file), chunktask represents SLURM_PROCID
        and chunks are distributed using SLURM_NTASKS as the modulo divisor.
        
        For local multiprocessing, chunktask represents local task ID and 
        chunks are distributed using self.nproc as the modulo divisor.
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
            chunksize = 0
        
        offset = hoffset
        
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
            
            if cdata.size == 0:
                if format_info.get('name') != "tipsy":
                    break
                return data
            
            # Progress reporting for tipsy format
            if format_info.get('name') == "tipsy":
                ncum = (offset - 32 + chunksize) // dsize
                gbread = (offset - 32 + chunksize) / 1024**3
                
                # Basic debug - always print from process 0 to verify we reach this code
                slurm_procid = int(os.environ.get('SLURM_PROCID', '0'))
                if chunk == 1 and slurm_procid == 0:
                    print(f"PROGRESS DEBUG: Reached tipsy progress section, chunk={chunk}, task_id={task_id}")
                    print(f"PROGRESS DEBUG: is_slurm_chunking={is_slurm_chunking}, _is_master_process()={_is_master_process()}")
                    sys.stdout.flush()
                
                # Debug: Always print first few chunks to verify we're getting here
                if chunk <= 3 and _is_master_process():
                    print(f"DEBUG: Chunk {chunk}, offset {offset}, chunksize {chunksize}, gbread {gbread:.2f}")
                
                # Progress reporting for tipsy format - show every 10th chunk to avoid spam
                if is_slurm_chunking and _is_master_process() and (chunk % 10 == 1 or chunk <= 3):
                    try:
                        file_size = filepath.stat().st_size if filepath.exists() else 100 * 1024**3
                        progress_pct = min(100, (offset + chunksize) / file_size * 100)
                        print(f"SLURM process {task_id}: {progress_pct:.1f}% complete, {gbread:.1f} GB read, {ncum:,} particles", flush=True)
                    except Exception as e:
                        print(f"SLURM process {task_id}: {gbread:.1f} GB read, {ncum:,} particles (error: {e})", flush=True)
                elif not is_slurm_chunking and self.verbose and chunk % 10 == 1:
                    print(f"Local task {task_id:>3d} read {ncum:>12} ({gbread:0.2f} GB)", flush=True)
                
                # Debug output (verbose mode only)
                if self.verbose and slurm_info['is_slurm']:
                    _log_info(f"DEBUG: SLURM process {slurm_info['procid']} chunk {chunk-1} at offset {offset}, read {chunksize} bytes")
                
                offset += chunksize
            
            # Process and add data
            data = DataProcessor.cull_tile_reshape(data, cdata, vars, bbox, r1, r2, 
                                                 format_info, lightcone)
        
        # No special completion handling needed for simple newline output
        
        return data

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
            lightcone: Whether in lightcone mode
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
        
        Implements the three-tier strategy based on file count:
        1. Many files (> SLURM_NTASKS): Distribute across SLURM processes + local multiprocessing
        2. Few files (≤ SLURM_NTASKS): One file per SLURM process  
        3. Single file: SLURM-level chunking
        
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
        available_files = []
        chunk = 0
        max_search = max(slurm_info['ntasks'] * slurm_info['cpus_per_task'], 100)
        
        while chunk < max_search:
            filepath = FileReader.get_filename(self.params.namespace, format_info, chunk, step)
            if filepath.exists():
                available_files.append(chunk)
                chunk += 1
            else:
                break
        
        n_files = len(available_files)
        ntasks = slurm_info['ntasks']
        procid = slurm_info['procid']
        cpus_per_task = slurm_info['cpus_per_task']
        
        if n_files == 0:
            _log_info("No TPS files found")
            return np.array([]).reshape(len(dprops['vars']), 0)
        
        elif n_files == 1:
            # SINGLE FILE: SLURM-level chunking (avoids multiprocessing + JAX conflicts)
            _log_info(f"Single TPS file: using SLURM-level chunking ({ntasks} processes)")
            result = self._read_step(step, bbox, dprops, format_info, False, redshift, procid)
            _log_info(f"SLURM process {procid}: I/O complete, {result.shape[1]:,} particles loaded")
            return result
        
        elif n_files <= ntasks:
            # FEW FILES: One complete file per SLURM process
            assigned_files = [f for i, f in enumerate(available_files) if i % ntasks == procid]
            
            if not assigned_files:
                _log_info(f"No files assigned to SLURM process {procid}")
                return np.array([]).reshape(len(dprops['vars']), 0)
            
            _log_info(f"Few files ({n_files}): SLURM process {procid} reading {len(assigned_files)} files")
            
            # Read assigned files sequentially (no multiprocessing needed)
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
            _log_info(f"Many files ({n_files}): SLURM process {procid} using {effective_nproc} cores for {len(assigned_files)} files")
            
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
        """Fetch particle data for the specified bounding box and parameters.
        
        Args:
            bbox: Bounding box [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
            dataset: Dataset type ('xvp', etc.)
            filetype: File format ('lcp', 'tps', 'fof')
            lightcone: Whether to use lightcone mode
            redshifts: List of redshifts to fetch (for non-lightcone mode)
            
        Returns:
            Data array or dictionary of data arrays
            
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
        
        with mp.Pool(processes=self.nproc) as pool:
            results = pool.starmap(self._read_step, args)
        
        concatenated = np.concatenate(results, axis=1)
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
            
            data[f"box{i}"] = np.rec.fromarrays(array.reshape(len(vars), -1), names=vars)
        
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
