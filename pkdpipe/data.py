import multiprocessing as mp
import numpy as np
from scipy.interpolate import interp1d
import argparse
import sys
import os
import re
import pkdpipe.dataspecs as pkds
from pkdpipe.cosmology import Cosmology

"""
pkdgrav3 data interface
"""

class Data:

    '''Data'''

    def _parse_param_file(self,content,varname,vartype=str):
        if vartype == str:
            match = re.search(r'(?<='+varname+r')\s*=\s*["\'].+["\']', content)
            if match is not None:
                return match[0].split("=")[-1].strip().replace('"', '').replace("'", "")
        elif vartype == float:
            match = re.search(r'(?<='+varname+r')\s*=\s*\d+\.\d+', content)
            if match is not None:
                return float(match[0].split("=")[-1].strip())
        elif vartype == int:
            match = re.search(r'(?<='+varname+r')\s*=\s*\d+', content)
            return   int(match[0].split("=")[-1].strip())
        return None

    def __init__(self, **kwargs):
        self.param_file = kwargs.get('param_file',None)
        self.nproc      = kwargs.get('nproc',1)
        self.params     = {}

        # read parameter file
        with open(self.param_file, "r") as f:
            content = f.read()

        # parse parameter file
        self.params['namespace'] = self._parse_param_file(content,  'achOutName',  str)
        self.params['zmax']      = self._parse_param_file(content,'dRedshiftLCP',float)
        self.params['boxsize']   = self._parse_param_file(content,    'dBoxSize',  int)
        self.params['nsteps']    = self._parse_param_file(content,      'nSteps',  int)
        self.params['omegam']    = self._parse_param_file(content,     'dOmega0',float)
        self.params['h']         = self._parse_param_file(content,           'h',float)
        self.params['ngrid']     = self._parse_param_file(content,       'nGrid',  int)

        self.params['zoutput']   = np.genfromtxt(self.params['namespace'] + ".log")[:, 1]

        # use interpolation for chi(z)
        zhigh = 1100.
        zlow  = 0.0
        nz    = 1000
        a     = np.logspace(np.log10(1/(1+zhigh)),np.log10(1/(1+zlow)),nz)
        z     = 1/a-1
        chi   = Cosmology(h = self.params['h'], omegam = self.params['omegam']).z2chi(z)
        self.chiofz = interp1d(z,chi)

    def _cull_shift_reshape(self,vars,data,shift,bounds):
        cdata = np.copy(data)
        dm = np.full(np.shape(cdata)[0],True)
        for var in ['x','y','z']:
           cdata[var] += shift[var]
        if 'x' in vars and 'y' in vars and 'z' in vars:
            dm = ((cdata['x']>bounds[0][0]) & (cdata['x']<=bounds[0][1]) &
                  (cdata['y']>bounds[1][0]) & (cdata['y']<=bounds[1][1]) &
                  (cdata['z']>bounds[2][0]) & (cdata['z']<=bounds[2][1]))
            if len(bounds) > 3:
                r = np.sqrt(cdata['x']**2+cdata['y']**2+cdata['z']**2)
                dm = (dm & (r>bounds[3][0]) & (r<=bounds[3][1]))
        return np.concatenate([cdata[var][dm] for var in vars]).reshape((len(vars),-1))

    def _cull_tile_reshape(self,data,cdata,vars,bbox,r1,r2,format,lightcone):
        shifts = []
        bounds = bbox.copy()
        if not format['sliced'] and lightcone:
            d=np.concatenate(np.meshgrid([-0.5,0.5],[-0.5,0.5],[-0.5,0.5])).reshape(3,8)
            for i in range(8):
                shifts.append(np.rec.fromarrays(d[:,i],names=['x','y','z']))
            bounds.append([r1,r2])
        else:
            shifts.append(np.rec.fromarrays(-np.asarray(format['offset']),names=['x','y','z']))

        for shift in shifts:
            sdata = self._cull_shift_reshape(vars,cdata,shift,bounds)
            data = np.concatenate((data,sdata),axis=1)
        return data

    def _get_filename(self,format,chunk,step):
        ext = format['ext']
        if format['ext'] is not None:
            return os.path.join(self.params['namespace'] + f".{step:05d}.{ext}.{chunk}")
        else:
            return os.path.join(self.params['namespace'] + f".{step:05d}")

    def _read_step(self,step,bbox,dprops,format,lightcone,redshift,chunktask):

        zvals = self.params['zoutput']
        z1 = zvals[step]
        z2 = zvals[step-1]
        vars  = dprops['vars']
        dtype = format['dtype']
        ext   = format['ext']

        chi1 = self.chiofz(z1) # Mpc/h
        chi2 = self.chiofz(z2) # Mpc/h

        r1 = chi1 / self.params['boxsize'] # internal length units
        r2 = chi2 / self.params['boxsize'] # internal length units

        data = np.concatenate([np.zeros(0,dtype=dptype) for dptype in dprops['dtype']]).reshape(len(vars),0)

        # return if redshift for this step outside of z1,z2 for lightcone, or not the step nearest to z otherwise
        if lightcone:
            if z1 > self.params['zmax']:
                return data
        else:
            if step != np.argmin(np.abs(redshift - zvals)):
                return data

        # iterate over chunks in step
        chunk = 0
        hoffset = format['hsize']
        dsize = format['dsize']
        count = -1
        if format['name'] == "tipsy":
            chunkmin = 2 * 1024**3
            count = chunkmin // dsize
            chunksize = count * dsize
        offset = hoffset
        moretoread = True
        while moretoread:

            # nothing to do if chunk not in chunktask
            if chunk % self.nproc != chunktask and chunktask >= 0:
                chunk += 1
                offset += chunksize
                continue

            # get the current file
            current_file = self._get_filename(format,chunk,step)
            chunk += 1

            # return if file doesn't exist
            if not os.path.exists(current_file):
                if format['name'] != "tipsy":
                    print(f"read step {step:>4}",end='\r')
                return data

            # read particles from file and add those within bounds to data
            cdata = np.fromfile(current_file, dtype=dtype, offset=offset, count=count)
            if format['name'] == "tipsy":
                if np.shape(cdata)[0] == 0:
                    return data
                ncum = (offset-32+chunksize)//dsize
                gbread = (offset-32+chunksize)/1024**3
                numkept = np.shape(data)[1]
                print(f"task {chunktask:>3d} read {ncum:>12} ({gbread:0.2f} GB)",end='\r')
                offset += chunksize
            elif np.shape(cdata)[0] == 0:
                moretoread = False
                continue
            else:
                print(f"step {step:>3d} read",end='\r')

            data = self._cull_tile_reshape(data,cdata,vars,bbox,r1,r2,format,lightcone)

        return data

    def fetch_data(self,bbox,dataset='xvp',filetype='lcp',lightcone=True,redshifts=[0]):

        dprops = pkds.properties[dataset]
        format = pkds.fileformats[filetype]
        vars   = dprops['vars']
        print("fetching data")

        data = None
        if lightcone:
            args = [[step,bbox,dprops,format,lightcone,0,-1] for step in range(1, self.params['nsteps'] + 1)]
            data = np.asarray([np.rec.fromarrays(np.concatenate(mp.Pool(processes=self.nproc).starmap(self._read_step,args),axis=1).reshape((len(vars),-1)),
                                      names=vars)])
        else:
            data = {}
            zout = []
            i = 0
            for redshift in redshifts:
                step = np.argmin(np.abs(redshift - self.params['zoutput']))
                zout.append(self.params['zoutput'][step])
                if filetype == 'fof':
                    array = self._read_step(step,bbox,dprops,format,lightcone,redshift,-1).reshape(len(vars),-1)
                elif filetype == 'tps':
                    args = [[step,bbox,dprops,format,lightcone,redshift,task] for task in range(self.nproc)]
                    array = np.concatenate(mp.Pool(processes=self.nproc).starmap(self._read_step,args),axis=1).reshape(len(vars),-1)
                data[f"box{i}"] = np.rec.fromarrays(array,names=vars)
                i+=1
            self.zout = np.asarray(zout)
            self.nout = i

        print("\ndata fetched")
        return data

    def matter_power(self,bbox,**kwargs):
        data = self.fetch_data(bbox,**kwargs)
        return data