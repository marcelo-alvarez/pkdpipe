"""
PKDGrav3 data interface - Refactored for better maintainability and robustness.
"""
import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d
import os
import re
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import pkdpipe.dataspecs as pkds  
from pkdpipe.cosmology import Cosmology


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    def parse_parameter(content: str, varname: str, vartype: type) -> Any:
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
    def load_parameters(param_file: Path) -> SimulationParams:
        """Load and validate simulation parameters from file.
        
        Args:
            param_file: Path to the parameter file
            
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
    def cull_shift_reshape(vars: List[str], data: np.ndarray, shift: np.ndarray,
                          bounds: List[List[float]]) -> np.ndarray:
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
    def cull_tile_reshape(data: np.ndarray, cdata: np.ndarray, vars: List[str], 
                         bbox: List[List[float]], r1: float, r2: float,
                         format_info: Dict[str, Any], lightcone: bool) -> np.ndarray:
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
    def get_filename(namespace: str, format_info: Dict[str, Any], chunk: int, step: int) -> Path:
        """Generate filename for a given chunk and step."""
        ext = format_info.get('ext')
        if ext is not None:
            filename = f"{namespace}.{step:05d}.{ext}.{chunk}"
        else:
            filename = f"{namespace}.{step:05d}"
        return Path(filename)
    
    @staticmethod
    def read_chunk(filepath: Path, dtype: np.dtype, offset: int = 0, count: int = -1) -> np.ndarray:
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

    def __init__(self, *, param_file: str, nproc: int = 1, verbose: bool = False):
        """Initialize the data interface.
        
        Args:
            param_file: Path to the PKDGrav3 parameter file
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
        
        logger.info(f"Initialized data interface for {self.params.namespace}")
        logger.info(f"Found {len(self.params.zoutput)} output redshifts")

    def _init_cosmology(self):
        """Initialize cosmological distance interpolation."""
        zhigh, zlow, nz = 1100.0, 0.0, 1000
        a = np.logspace(np.log10(1 / (1 + zhigh)), np.log10(1 / (1 + zlow)), nz)
        z = 1 / a - 1
        
        cosmology = Cosmology(h=self.params.h, omegam=self.params.omegam)
        chi = cosmology.z2chi(z)
        self.chiofz = interp1d(z, chi, kind='linear', fill_value='extrapolate')

    def _read_step(self, step: int, bbox: List[List[float]], dprops: Dict[str, Any], 
                   format_info: Dict[str, Any], lightcone: bool, redshift: float, 
                   chunktask: int) -> np.ndarray:
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

    def _process_chunks(self, step: int, bbox: List[List[float]], vars: List[str], 
                       dtype: np.dtype, format_info: Dict[str, Any], lightcone: bool,
                       r1: float, r2: float, chunktask: int, data: np.ndarray) -> np.ndarray:
        """Process all chunks for a given step."""
        chunk = 0
        hoffset = format_info.get('hsize', 0)
        dsize = format_info.get('dsize', 1)
        count = -1
        
        # Configure chunking for tipsy format
        if format_info.get('name') == "tipsy":
            chunkmin = 2 * 1024**3
            count = chunkmin // dsize
            chunksize = count * dsize
        else:
            chunksize = 0
        
        offset = hoffset
        
        while True:
            # Skip chunks not assigned to this task
            if chunktask >= 0 and chunk % self.nproc != chunktask:
                chunk += 1
                offset += chunksize
                continue
            
            # Get current file
            filepath = FileReader.get_filename(self.params.namespace, format_info, chunk, step)
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
                if self.verbose:
                    print(f"task {chunktask:>3d} read {ncum:>12} ({gbread:0.2f} GB)", end='\r')
                offset += chunksize
            
            # Process and add data
            data = DataProcessor.cull_tile_reshape(data, cdata, vars, bbox, r1, r2, 
                                                 format_info, lightcone)
        
        return data

    def fetch_data(self, bbox: List[List[float]], dataset: str = 'xvp', 
                   filetype: str = 'lcp', lightcone: bool = False, 
                   redshifts: List[float] = [0]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
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
        
        logger.info(f"Fetching {dataset} data with {filetype} format")
        logger.info(f"Lightcone: {lightcone}, Variables: {vars}")
        logger.info(f"Bounding box: {bbox}")
        
        if lightcone:
            return self._fetch_lightcone_data(bbox, dprops, format_info, vars)
        else:
            return self._fetch_snapshot_data(bbox, dprops, format_info, vars, 
                                           filetype, redshifts)

    def _fetch_lightcone_data(self, bbox: List[List[float]], dprops: Dict[str, Any], 
                             format_info: Dict[str, Any], vars: List[str]) -> np.ndarray:
        """Fetch data in lightcone mode."""
        max_step = len(self.params.zoutput) - 1
        args = [[step, bbox, dprops, format_info, True, 0, -1] 
                for step in range(1, max_step + 1)]
        
        with mp.Pool(processes=self.nproc) as pool:
            results = pool.starmap(self._read_step, args)
        
        concatenated = np.concatenate(results, axis=1)
        # Wrap in array to maintain backward compatibility with legacy format (1, N) shape
        data = np.asarray([np.rec.fromarrays(concatenated, names=vars)])
        
        logger.info(f"Lightcone data shape: {data.shape}")
        return data

    def _fetch_snapshot_data(self, bbox: List[List[float]], dprops: Dict[str, Any], 
                            format_info: Dict[str, Any], vars: List[str], 
                            filetype: str, redshifts: List[float]) -> Dict[str, np.ndarray]:
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
                args = [[step, bbox, dprops, format_info, False, redshift, task] 
                       for task in range(self.nproc)]
                with mp.Pool(processes=self.nproc) as pool:
                    results = pool.starmap(self._read_step, args)
                array = np.concatenate(results, axis=1)
            else:
                raise ValueError(f"Unsupported filetype for snapshots: {filetype}")
            
            data[f"box{i}"] = np.rec.fromarrays(array.reshape(len(vars), -1), names=vars)
        
        self.zout = np.array(zout)
        self.nout = len(data)
        
        logger.info(f"Snapshot data: {len(data)} boxes")
        for key, value in data.items():
            logger.info(f"  {key}: shape {value.shape}")
        
        return data

    def matter_power(self, bbox: List[List[float]], ngrid: int = 256, n_devices: int = 1, 
                    subtract_shot_noise: bool = False, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
