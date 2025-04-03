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

def approximate_geometry(omegam,h):
    from scipy.interpolate import interp1d

    c = 3e5

    H0 = 100*h
    nz = 10000
    z1 = 0.0
    z2 = 20.0
    za = np.linspace(z1,z2,nz)
    dz = za[1]-za[0]

    H      = lambda z: H0*np.sqrt(omegam*(1+z)**3+1-omegam)
    dchidz = lambda z: c/H(z)

    chia = np.cumsum(dchidz(za))*dz*h # Mpc/h
    chia -= chia[0]

    zofchi = interp1d(chia,za)
    chiofz = interp1d(za,chia)

    return zofchi,chiofz

dataproperties = {
    'xvp' : {'dtype' : ['f4','f4','f4','f4','f4','f4'], 
              'vars' : [ 'x', 'y', 'z','vx','vy','vz']},

    'xvh' : {'dtype' : ['f4','f4','f4','f4','f4','f4',   'i4'], 
              'vars' : [ 'x', 'y', 'z','vx','vy','vz','npart']},
}

fileformats = {
    # lcp format
    #      all - 0,...,9 (36 bytes)
    #     mass - 0,1     (1 double)
    #    x,y,z - 2,3,4   (3 floats)
    # vx,vy,vz - 5,6,7   (3 floats)
    #      eps - 8       (1 float)
    #      phi - 9       (1 float)   
    'lcp' : {'dtype'  : [('mass','d'),("x",'f4'),("y",'f4'),("z",'f4'),
                        ("vx",'f4'),("vy",'f4'),("vz",'f4'),("eps",'f4'),("phi",'f4')],
             'ext'    : 'lcp',
             'name'   : 'lightcone',
             'sliced' : True,
             'offset' : [0,0,0]},
    # fof format
    #      all - 0,...,33 (132 bytes)
    #    x,y,z - 0,1,2    (3 floats)
    #      pot - 3        (1 float)
    #     dum1 - 4-8      (12 bytes)
    # vx,vy,vz - 7,8,9    (3 floats)
    #     dum2 - 4-8      (84 bytes)
    #    npart - 31       (1 int)
    #     dum3 - 32,33    (8 bytes)
    'fof' : {'dtype'  : [ ('x','f4'), ('y','f4'), ('z','f4'),('pot','f4'),('dum1',('f4',3)),
                        ('vx','f4'),('vy','f4'),('vz','f4'),('dum2',('f4',21)),
                        ('npart','i4'),('dum3',('f4',2))],
             'ext'    : 'fofstats',
             'name'   : 'fof',
             'sliced' : False,
             'offset' : [-0.5,-0.5,-0.5]}
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
        self.params['omegam']    = self._parse_param_file(content,     'dOmega0',float)
        self.params['h']         = self._parse_param_file(content,           'h',float)
        self.params['ngrid']     = self._parse_param_file(content,       'nGrid',  int)

        self.params['zoutput']   = np.genfromtxt(self.params['namespace'] + ".log")[:, 1]

        self.zofchi, self.chiofz = approximate_geometry(self.params['omegam'],self.params['h'])

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

    def _read_step(self,step,bbox,dprops,format,lightcone,redshift):

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

        # iterate over processes with files
        current_proc = 0
        while True:
            # get the current file
            current_file = os.path.join(self.params['namespace'] + f".%05d.{ext}.%i" % (step, current_proc))
            current_proc += 1

            # return if file doesn't exist
            if not os.path.exists(current_file):
                print(f"read step {step:>4}",end='\r')
                return data

            # read particles from file and add those within bounds to data 
            cdata = np.fromfile(current_file, dtype=dtype)
            if np.shape(cdata)[0] == 0:
                continue

            data = self._cull_tile_reshape(data,cdata,vars,bbox,r1,r2,format,lightcone)

    def fetch_data(self,bbox,dataset='xv',filetype='lcp',lightcone=True,redshifts=[0]):

        dprops = dataproperties[dataset]
        format = fileformats[filetype]
        vars   = dprops['vars']
        print("fetching data")

        if lightcone:
            args = [[step,bbox,dprops,format,lightcone,0] for step in range(1, self.params['nsteps'] + 1)]
            data = np.asarray([np.rec.fromarrays(np.concatenate(mp.Pool(processes=self.nproc).starmap(self._read_step,args),axis=1).reshape((len(vars),-1)),
                                      names=vars)])
        else:
            data = {}
            zout = []
            i = 0
            for redshift in redshifts:
                step = np.argmin(np.abs(redshift - self.params['zoutput']))
                data[f"box{i}"] = (np.rec.fromarrays(self._read_step(step,bbox,dprops,format,lightcone,redshift).reshape(len(vars),-1),names=vars))
                zout.append(self.params['zoutput'][step])
                i+=1
            self.zout = np.asarray(zout)
            self.nout = i
        print("\ndata fetched")
        return data
