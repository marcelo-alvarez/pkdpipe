import numpy as np
import sys
import pathlib
import subprocess
import os
import shutil
from string import Template
from pkdpipe.cosmology import Cosmology
from pkdpipe.cli import parsecommandline

"""
pkdpipe simulation
"""

# Defaults
dspath  = f"{os.getenv('SCRATCH')}/pkdgrav3/runs"
dpath   = f"{os.getenv('CFS')}/cosmosim/slac/{os.getenv('USER')}/pkdgrav3/runs"

djobname   = "N{nGrid}-L{dBoxSize}-{nodes.gpupern}gpus"
demail     = "${pkdgravemail}"
dsimname   = "lcone-small"
dcosmo     = "desi-dr2-planck-act-mnufree"

dredshiftlist = sorted([3., 2.81388253, 2.63504180, 2.46500347, 2.30360093,
    2.14960630, 2.00300300, 1.86204923, 1.72851296, 1.60145682, 1.48015873,
    1.36406619, 1.25377507, 1.14868930, 1.04834084, 0.95274360, 0.86150410,
    0.77462289, 0.69176112, 0.61290323, 0.53751538, 0.46584579, 0.39742873, 
    0.33226752, 0.27000254, 0.21065375, 0.15420129, 0.10035211, 0.04898773, 
    0.00000000] + 
    [0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.33, 0.922, 0.955],reverse=True)

ddRedTo = f'{[f'{z:0.4f}' for z in dredshiftlist]}'.replace("'", "")
dnSteps = f'{[f'{i}' for i in range(len(dredshiftlist))]}'.replace("'", "")

templatedir = f"{str(pathlib.Path(__file__).parent.resolve())}/../templates"
scriptdir   = f"{str(pathlib.Path(__file__).parent.resolve())}/../scripts"

partemp   = f"{templatedir}/lightcone.par"
slurmtemp = f"{templatedir}/slurm.sh"
runtemp   = f"{scriptdir}/run.sh"

def safemkdir(dir):
    if os.path.isdir(dir):
        response = input(f"{dir} already exists; enter REMOVE if you want to delete it\n")
        if response == "REMOVE":
            nsleep=10
            for i in range(nsleep):
                print(f"will delete {dir} in {nsleep-i} seconds...",end='\r')
                subprocess.call(f"sleep 1", shell=True)
            print()
            if len(dir) > 10:
                subprocess.call(f"rm -rf {dir}", shell=True)
            else:
                return 1
        else:
            return 1
    # make directory
    os.makedirs(f"{dir}")
    return 0

def copytemplate(templatefile,outfile,data):
    with open(templatefile, "r") as file:
        templines = file.read()
    template = Template(templines)
    lines = template.substitute(data)
    with open(outfile, 'w') as par:
        par.write(lines)

def get_simulation(simname):
    pkp_params = {
        'sbatch'  : {'val' :      False, 'type' :  bool, 'desc' : 'submit with sbatch; otherwise only create dir & files'},
        'interact': {'val' :      False, 'type' :  bool, 'desc' : 'submit as interactive; otherwise only create dir & files'}
    }
    job_params = {
        'nodes'   : {'val' :          2, 'type' :   int, 'desc' : 'number of nodes'},
        'cpupert' : {'val' :        128, 'type' :   int, 'desc' : 'CPUs per task'},
        'gpupern' : {'val' :          4, 'type' :   int, 'desc' : 'GPUs per node'},
        'rundir'  : {'val' :      dpath, 'type' :   str, 'desc' : 'path for runs'},
        'scrdir'  : {'val' :     dspath, 'type' :   str, 'desc' : 'scratch path for runs (if --scratch provided)'},
        'jobname' : {'val' :   djobname, 'type' :   str, 'desc' : 'directory name for run'},
        'email'   : {'val' :     demail, 'type' :   str, 'desc' : 'email for slurm notifications'},
        'tlimit'  : {'val' : '48:00:00', 'type' :   str, 'desc' : 'time limit'},
        'simname' : {'val' :   dsimname, 'type' :   str, 'desc' : 'simulation parameter set name'},
        'cosmo'   : {'val' :     dcosmo, 'type' :   str, 'desc' : 'cosmology'},
        'scratch' : {'val' :      False, 'type' :  bool, 'desc' : 'use scratch for output'}
    }
    pkd_params = {
        'nGrid'        : {'val' :       1400, 'type' :   int, 'desc' : 'cube root particle number'},
        'dBoxSize'     : {'val' :       1050, 'type' :   int, 'desc' : 'boxsize [Mpc/h]'},
        'dRedFrom'     : {'val' :         12, 'type' : float, 'desc' : 'starting redshift'},
        'iLPT'         : {'val' :          3, 'type' :   int, 'desc' : 'LPT order for ICs'},
        'dRedTo'       : {'val' :    ddRedTo, 'type' :   str, 'desc' : 'output redshifts'},
        'nSteps'       : {'val' :    dnSteps, 'type' :   str, 'desc' : 'output steps'},
        'iOutInterval' : {'val' :          1, 'type' :   int, 'desc' : 'snapshot interval'}
    }
    params = pkp_params | job_params | pkd_params

    # only lightcone tests are implemented so far
    if simname == 'lcone-medium':
        pass
    elif simname == 'lcone-small':
        params['dBoxSize']['val']  = 525
        params['nGrid']['val'] = 700
        params['nodes']['val'] = 1
    elif simname == 'lcone-large':
        params['dBoxSize']['val']  = 2100
        params['nGrid']['val'] = 2800
        params['nodes']['val'] = 16
    return params

class Simulation:

    '''Simulation'''
    def __init__(self, **kwargs):

        self.simname = kwargs.get('simname', dsimname)

        # initialize parameters using parser
        self.params = parsecommandline(get_simulation(self.simname),
                                       description='Commandline interface to pkdpipe.Simulation')

        if kwargs.get('parse', False):
            # if requested parse again to retrieve updated simulation name from command-line
            self.simname = self.params['simname']
            self.params = parsecommandline(get_simulation(self.simname),
                                           description='Commandline interface to pkdpipe.Simulation')

        # update parameters via kwargs
        for key in self.params:
            self.params[key] = kwargs.get(key,self.params[key])

    def create(self):

        # file and directory names
        jobdir    = f"{self.params['rundir']}/{ self.params['jobname']}"
        jobscr    = f"{self.params['scrdir']}/{ self.params['jobname']}"
        achTfFile = f"{jobdir}/{self.params['jobname']}.transfer"
        slurmfile = f"{jobdir}/{self.params['jobname']}.sbatch"
        parfile   = f"{jobdir}/{self.params['jobname']}.par"
        logfile   = f"{jobdir}/{self.params['jobname']}.log"
        runscript = f"{jobdir}/run.sh"

        # run command
        runcmd = f"{runscript} {parfile} {logfile}"

        # set job name
        if self.params['jobname'] == djobname:
            gpus     = self.params['gpupern'] * self.params['nodes']
            nGrid    = self.params['nGrid']
            dBoxSize = self.params['dBoxSize']
            self.params['jobname'] = f"N{nGrid:04d}-L{dBoxSize}-{gpus:03d}gpus"
            
        # set email
        if self.params['email'] == demail:
            self.params['email'] = os.getenv('pkdgravemail')

        # make job directory
        if safemkdir(jobdir) > 0:
            print('exiting...')
            sys.exit(1)

        # make scratch output directory and link
        if self.params['scratch']:
            outdir=f"{jobscr}/output"
            if safemkdir(outdir) > 0:
                print('exiting...')
                sys.exit(1)
            subprocess.call(f"ln -s {outdir} {jobdir}/", shell=True)

        # copy runscript
        shutil.copy(runtemp, runscript)

        # get cosmology
        cosmo  = Cosmology(cosmology=self.params['cosmo'])

        # get maximum redshift
        dRedshiftLCP = f"{cosmo.chi2z(self.params['dBoxSize']/cosmo.params['h']*0.98):0.2f}"

        # write transfer file
        cosmo.writetransfer(achTfFile)

        # generate parameter file
        copytemplate(partemp,parfile,{
                    "achOutName"  : jobdir,
                    "achTfFile"   : achTfFile,
                    "dRedshiftLCP": dRedshiftLCP,
                    "dSpectral"   : f"{cosmo.params['ns']:0.6f}",
                    "h"           : f"{cosmo.params['h']:0.6f}",
                    "dOmega0"     : f"{cosmo.params['omegam']:0.6f}",
                    "dLambda"     : f"{cosmo.params['omegal']:0.6f}",
                    "dSigma8"     : f"{cosmo.params['sigma8']:0.6f}",
                    "dRedFrom"    : f"{self.params['dRedFrom']}",
                    "nGrid"       : f"{self.params['nGrid']:<4}",
                    "dBoxSize"    : f"{self.params['dBoxSize']:<4}",
                    "iLPT"        : f"{self.params['iLPT']}",
                    "dRedTo"      : f"{self.params['dRedTo']}",
                    "nSteps"      : f"{self.params['nSteps']}",
                    "iOutInterval": f"{self.params['iOutInterval']}"
        })

        # generate slurm batch script
        copytemplate(slurmtemp,slurmfile,{
                    "tlimit"    : self.params['tlimit'],
                    "email"     : self.params['email'],
                    "nodes"     : self.params['nodes'],
                    "cpupert"   : self.params['cpupert'],
                    "gpupern"   : self.params['gpupern'],
                    "jobname"   : self.params['jobname'],
                    "runcmd"    : runcmd
            })
        print(f"created {slurmfile}")

        # submit slurm batch job
        if self.params['sbatch']:
            subprocess.call(f"sbatch {slurmfile}", shell=True, cwd=jobdir)

        # run interactive job
        if self.params['interact']:
            runcmd = f"{runcmd} interactive"#.split(" ")
            subprocess.call(runcmd, shell=True, cwd=jobdir)