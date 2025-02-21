import multiprocessing as mp
import numpy as np
import argparse
import sys
import os
import re
import numpy.lib.recfunctions as recfc

"""
pkdgrav3 data interface
"""

dataproperties = {
    'xv' : {'dtype' : ['f','f','f', 'f', 'f', 'f'], 
            'vars'  : ['x','y','z','vx','vy','vz']}
}

fileformats = {
    'lcp' : {'dtype' : [('mass','d'),("x",'f'),("y",'f'),("z",'f'),
                        ("vx",'f'),("vy",'f'),("vz",'f'),("eps",'f'),("phi",'f')],
             'ext'   : 'lcp'}
}

class DataInterface:

    '''DataInterface'''

    def _parse_param_file(self,content,varname,vartype=str):
        if vartype == str:
            return       re.search(r'(?<='+varname+r')\s*=\s*["\'].+["\']', content)[0].split("=")[-1].strip().replace('"', '').replace("'", "")
        elif vartype == float:
            return float(re.search(r'(?<='+varname+r')\s*=\s*\d+\.\d+',     content)[0].split("=")[-1].strip())
        elif vartype == int:
            return   int(re.search(r'(?<='+varname+r')\s*=\s*\d+',          content)[0].split("=")[-1].strip())
        else:
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
        self.params['zoutput']   = np.genfromtxt(self.params['namespace'] + ".log")[:, 1]

    def _read_step(self,step,bounds,dprops,format):

        z = self.params['zoutput'][step]
        vars  = dprops['vars']
        dtype = format['dtype']
        ext   = format['ext']
        data = np.concatenate([np.zeros(0,dtype=dptype) for dptype in dprops['dtype']]).reshape(len(vars),0)

        # return if redshift for this step outside of lightcone
        if z > self.params['zmax']:
            return data

        # iterate over processes with files
        current_proc = 0
        while True:
            # get the current file
            current_file = os.path.join(self.params['namespace'] + f".%05d.{ext}.%i" % (step, current_proc))
            current_proc += 1

            # return if file doesn't exist
            if not os.path.exists(current_file):
                print(f"completed step {step:>4}",end='\r')
                return data

            # read particles from file and add those within bounds to data 
            cdata = np.fromfile(current_file, dtype=dtype)
            if np.shape(cdata)[0] == 0:
                continue
            dm = np.full(np.shape(cdata)[0],True)

            if 'x' in vars and 'y' in vars and 'z' in vars:
                dm = ((cdata['x']>bounds[0][0]) & (cdata['x']<bounds[0][1]) &
                    (cdata['y']>bounds[1][0]) & (cdata['y']<bounds[1][1]) &
                    (cdata['z']>bounds[2][0]) & (cdata['z']<bounds[2][1]))
            cdata = np.concatenate([cdata[var][dm] for var in vars]).reshape((len(vars),-1))
            data = np.concatenate((data,cdata),axis=1)

    def fetch_data(self,bounds,dataset='xv',filetype='lcp'):

        dprops = dataproperties[dataset]
        format = fileformats[filetype]
        vars   = dprops['vars']
        args = [[step,bounds,dprops,format] for step in range(1, self.params['nsteps'] + 1)]

        print()

        return np.rec.fromarrays(np.concatenate(mp.Pool(processes=self.nproc).starmap(self._read_step,args),axis=1).reshape((len(vars),-1)),
                                  names=vars)

