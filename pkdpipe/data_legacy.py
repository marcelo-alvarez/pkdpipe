import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d
import argparse
import sys
import os
import re
from typing import Dict, List, Any

import pkdpipe.dataspecs as pkds  
from pkdpipe.cosmology import Cosmology


class Data:
    """pkdgrav3 data interface"""

    def __init__(self, *, param_file: str, nproc: int = 1):
        self.param_file = param_file
        self.nproc = nproc
        self.params = {}

        # Read parameter file
        with open(self.param_file, "r") as f:
            content = f.read()

        # Parse parameter file
        self.params['namespace'] = self._parse_param_file(content, 'achOutName', str)
        self.params['zmax'] = self._parse_param_file(content, 'dRedshiftLCP', float)
        self.params['boxsize'] = self._parse_param_file(content, 'dBoxSize', int) 
        self.params['nsteps'] = self._parse_param_file(content, 'nSteps', int)
        self.params['omegam'] = self._parse_param_file(content, 'dOmega0', float)
        self.params['h'] = self._parse_param_file(content, 'h', float)
        self.params['ngrid'] = self._parse_param_file(content, 'nGrid', int)

        self.params['zoutput'] = np.genfromtxt(self.params['namespace'] + ".log")[:, 1]

        # Use interpolation for chi(z)
        zhigh = 1100.0
        zlow = 0.0  
        nz = 1000
        a = np.logspace(np.log10(1 / (1 + zhigh)), np.log10(1 / (1 + zlow)), nz)
        z = 1 / a - 1
        chi = Cosmology(h=self.params['h'], omegam=self.params['omegam']).z2chi(z)
        self.chiofz = interp1d(z, chi)

    def _parse_param_file(self, content: str, varname: str, vartype: type) -> Any:
        """Parse a variable from a parameter file."""
        if vartype == str:
            match = re.search(rf'{re.escape(varname)}\s*=\s*["\'"]([^"\']+)["\'"]', content)
            if match is not None:
                return match.group(1)
        elif vartype == float:
            match = re.search(rf'{re.escape(varname)}\s*=\s*(\d+\.?\d*)', content)
            if match is not None:
                return float(match.group(1))
        elif vartype == int:
            # First try to match a list of integers [1, 2, 3, ...]
            list_match = re.search(rf'{re.escape(varname)}\s*=\s*\[([^\]]+)\]', content)
            if list_match is not None:
                # Extract the list content and parse the integers
                list_content = list_match.group(1)
                # Split by comma and convert to integers
                numbers = [int(x.strip()) for x in list_content.split(',') if x.strip()]
                return sum(numbers)
            
            # If no list found, try to match a single integer
            match = re.search(rf'{re.escape(varname)}\s*=\s*(\d+)', content)
            if match is not None:
                return int(match.group(1))
        return None

    def _cull_shift_reshape(self, vars: List[str], data: np.ndarray, shift: np.ndarray,
                            bounds: List[List[float]]) -> np.ndarray:
        """Cull, shift, and reshape data based on variables and bounds."""
        cdata = np.copy(data)
        dm = np.full(cdata.shape[0], True)
        for var in ['x', 'y', 'z']:
            cdata[var] += shift[var]
        if 'x' in vars and 'y' in vars and 'z' in vars:
            dm = ((cdata['x'] > bounds[0][0]) & (cdata['x'] <= bounds[0][1]) &
                  (cdata['y'] > bounds[1][0]) & (cdata['y'] <= bounds[1][1]) &
                  (cdata['z'] > bounds[2][0]) & (cdata['z'] <= bounds[2][1]))
            if len(bounds) > 3:
                r = np.sqrt(cdata['x'] ** 2 + cdata['y'] ** 2 + cdata['z'] ** 2)
                dm = (dm & (r > bounds[3][0]) & (r <= bounds[3][1]))
        return np.concatenate([cdata[var][dm] for var in vars]).reshape((len(vars), -1))

    def _cull_tile_reshape(self, data: np.ndarray, cdata: np.ndarray, vars: List[str], 
                           bbox: List[List[float]], r1: float, r2: float,
                           format: Dict[str, Any], lightcone: bool) -> np.ndarray:
        """Cull, tile, and reshape data."""
        shifts = []
        bounds = bbox.copy()
        if not format['sliced'] and lightcone:
            d = np.concatenate(np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])).reshape(3, 8)
            for i in range(8):
                shifts.append(np.rec.fromarrays(d[:, i], names=['x', 'y', 'z']))
            bounds.append([r1, r2])
        else:
            shifts.append(np.rec.fromarrays(-np.asarray(format['offset']), names=['x', 'y', 'z']))

        for shift in shifts:
            sdata = self._cull_shift_reshape(vars, cdata, shift, bounds)
            data = np.concatenate((data, sdata), axis=1)
        return data

    def _get_filename(self, format: Dict[str, Any], chunk: int, step: int) -> str:
        """Get the filename for a given chunk and step."""
        ext = format['ext']
        if format['ext'] is not None:
            return os.path.join(self.params['namespace'] + f".{step:05d}.{ext}.{chunk}")
        else:
            return os.path.join(self.params['namespace'] + f".{step:05d}")

    def _read_step(self, step: int, bbox: List[List[float]], dprops: Dict[str, Any], 
                   format: Dict[str, Any], lightcone: bool, redshift: float, 
                   chunktask: int) -> np.ndarray:
        """Read data for a single time step."""
        zvals = self.params['zoutput']
        z1 = zvals[step]
        z2 = zvals[step - 1]
        vars = dprops['vars']
        dtype = format['dtype']
        ext = format['ext']

        chi1 = self.chiofz(z1)  # Mpc/h
        chi2 = self.chiofz(z2)  # Mpc/h

        r1 = chi1 / self.params['boxsize']  # internal length units
        r2 = chi2 / self.params['boxsize']  # internal length units

        data = np.concatenate([np.zeros(0, dtype=dptype) for dptype in dprops['dtype']]).reshape(len(vars), 0)

        # Return if redshift for this step outside of z1,z2 for lightcone, or not the step nearest to z otherwise
        if lightcone:
            if z1 > self.params['zmax']:
                return data
        else:
            if step != np.argmin(np.abs(redshift - zvals)):
                return data

        # Iterate over chunks in step
        chunk = 0
        hoffset = format['hsize']
        dsize = format['dsize']
        count = -1
        if format['name'] == "tipsy":
            chunkmin = 2 * 1024 ** 3
            count = chunkmin // dsize
            chunksize = count * dsize
        offset = hoffset
        moretoread = True
        files_found = 0
        particles_read = 0
        
        while moretoread:

            # Nothing to do if chunk not in chunktask
            if chunk % self.nproc != chunktask and chunktask >= 0:
                chunk += 1
                offset += chunksize
                continue

            # Get the current file
            current_file = self._get_filename(format, chunk, step)
            chunk += 1

            # Return if file doesn't exist
            if not os.path.exists(current_file):
                if format['name'] != "tipsy":
                    # Only print progress for tipsy format to avoid spam
                    pass
                break
            
            files_found += 1

            # Read particles from file and add those within bounds to data
            cdata = np.fromfile(current_file, dtype=dtype, offset=offset, count=count)
            particles_read += len(cdata)
            
            if format['name'] == "tipsy":
                if cdata.shape[0] == 0:
                    return data
                ncum = (offset - 32 + chunksize) // dsize
                gbread = (offset - 32 + chunksize) / 1024 ** 3
                numkept = data.shape[1]
                print(f"task {chunktask:>3d} read {ncum:>12} ({gbread:0.2f} GB)", end='\r')
                offset += chunksize
            elif cdata.shape[0] == 0:
                moretoread = False
                continue
            else:
                print(f"step {step:>3d} read", end='\r')

            data = self._cull_tile_reshape(data, cdata, vars, bbox, r1, r2, format, lightcone)

        # Only print final summary to avoid multiprocessing spam
        if chunktask <= 0:  # Only print from main process or when chunktask is -1
            print(f"Step {step}: Found {files_found} files, read {particles_read} particles, final data shape: {data.shape}")
        return data

    def fetch_data(self, bbox: List[List[float]], dataset: str = 'xvp', filetype: str = 'lcp',
                   lightcone: bool = True, redshifts: List[float] = [0]) -> np.ndarray:
        """Fetch data for the given bounding box and parameters."""
        dprops = pkds.properties[dataset]
        format = pkds.fileformats[filetype]
        vars = dprops['vars']
        print("fetching data")
        print(f"Dataset: {dataset}, Filetype: {filetype}, Lightcone: {lightcone}")
        print(f"Variables: {vars}")
        print(f"Bounding box: {bbox}")

        data = None
        if lightcone:
            # Use the actual number of output steps from zvals instead of nsteps
            # Note: step starts from 1 since we need zvals[step-1] for z2
            max_step = len(self.params['zoutput']) - 1  # -1 because step can go up to max_step (0-indexed)
            print(f"Processing {max_step} steps from 1 to {max_step}")
            args = [[step, bbox, dprops, format, lightcone, 0, -1] for step in range(1, max_step + 1)]
            result = mp.Pool(processes=self.nproc).starmap(self._read_step, args)
            concatenated = np.concatenate(result, axis=1)
            print(f"Concatenated data shape: {concatenated.shape}")
            data = np.asarray([np.rec.fromarrays(concatenated.reshape((len(vars), -1)), names=vars)])
        else:
            data = {}
            zout = []
            i = 0
            for redshift in redshifts:
                step = np.argmin(np.abs(redshift - self.params['zoutput']))
                zout.append(self.params['zoutput'][step])
                if filetype == 'fof':
                    array = self._read_step(step, bbox, dprops, format, lightcone, redshift, -1).reshape(len(vars), -1)
                elif filetype == 'tps':
                    args = [[step, bbox, dprops, format, lightcone, redshift, task] for task in range(self.nproc)]
                    array = np.concatenate(mp.Pool(processes=self.nproc).starmap(self._read_step, args), axis=1).reshape(len(vars), -1)
                data[f"box{i}"] = np.rec.fromarrays(array, names=vars)
                i += 1
            self.zout = np.asarray(zout)
            self.nout = i

        if lightcone:
            print(f"\ndata fetched, final shape: {data.shape}")
        else:
            print(f"\ndata fetched, {len(data)} boxes")
            for key, value in data.items():
                print(f"  {key}: shape {value.shape}")
        return data

    def matter_power(self, bbox: List[List[float]], **kwargs) -> np.ndarray:
        """Compute the matter power spectrum for the given bounding box."""
        data = self.fetch_data(bbox, **kwargs)
        return data