import os
import time
import shutil
import subprocess
from string import Template
from typing import Dict, Any, Union, Optional
import pathlib

from .cosmology import Cosmology
from .cli import parsecommandline
from .parameter_types import extract_cosmological_parameters
from .config import (
    DEFAULT_RUN_DIR_BASE,
    DEFAULT_SCRATCH_DIR_BASE,
    DEFAULT_JOB_NAME_TEMPLATE,
    DEFAULT_SIMULATION_NAME,
    DEFAULT_COSMOLOGY_NAME,
    DEFAULT_REDSHIFT_TARGETS_STR,
    DEFAULT_NSTEPS_STR,
    SIMULATION_PRESETS,
    TEMPLATE_LIGHTCONE_PAR,
    TEMPLATE_SLURM_SH,
    SCRIPT_RUN_SH,
    DEFAULT_TIME_LIMIT,
    DEFAULT_NODES,
    DEFAULT_CPUS_PER_TASK,
    DEFAULT_GPUS_PER_NODE,
    DEFAULT_NGRID,
    DEFAULT_BOXSIZE_MPC_H,
    DEFAULT_REDSHIFT_FROM,
    DEFAULT_LPT_ORDER,
    DEFAULT_OUTPUT_INTERVAL,
    DirectoryError,
    TemplateError,
    JobSubmissionError,
    PKDGRAVEMAIL
)

"""
pkdpipe simulation setup and management module.

This module provides the `Simulation` class, which is responsible for orchestrating
the setup of pkdgrav3 N-body simulations. This includes:
- Managing simulation parameters, with support for presets and command-line overrides.
- Creating necessary directory structures for simulation runs (run and scratch directories).
- Generating configuration files required by pkdgrav3:
    - Transfer function file (via the `Cosmology` class).
    - Main parameter file (.par) for pkdgrav3.
    - SLURM batch submission script (.sbatch).
- Copying essential scripts (e.g., `run.sh`) to the job directory.
- Optionally submitting the job to a SLURM scheduler or providing instructions for manual execution.

The module also includes utility functions for safe directory creation (`safemkdir`)
and template file processing (`copy_template_with_substitution`).
"""

def safemkdir(dir_path: str, force_remove_delay: int = 10) -> None:
    """
    Safely creates a directory. If the directory already exists, it prompts
    the user for confirmation before attempting to remove and recreate it.

    This function is designed to prevent accidental deletion of important directories.
    A delay is enforced before removal if confirmed by the user.

    Args:
        dir_path (str): The absolute or relative path to the directory to be created.
        force_remove_delay (int): The number of seconds to wait before removing the
                                  existing directory if the user confirms removal.
                                  Defaults to 10 seconds.

    Raises:
        DirectoryError: If directory creation fails, if the user aborts due to an
                        existing directory, or if removal of an existing directory fails.
                        Also raised if the path is deemed too shallow (e.g., '/') for
                        automatic removal as a safety measure.
    """
    if os.path.isdir(dir_path):
        response = input(
            f"'{dir_path}' already exists. Enter 'REMOVE' to delete it and continue, "
            f"or anything else to abort: "
        )
        if response == "REMOVE":
            print(f"Attempting to remove '{dir_path}' in {force_remove_delay} seconds...")
            try:
                for i in range(force_remove_delay, 0, -1):
                    print(f"Removing in {i}...", end='\r')
                    time.sleep(1)
                print("Removing now...          ")
                
                if len(pathlib.Path(dir_path).parts) < 3 and "*" not in dir_path:
                    raise DirectoryError(f"Path '{dir_path}' is too shallow for automatic removal for safety reasons.")
                
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' removed successfully.")
            except Exception as e:
                raise DirectoryError(f"Failed to remove existing directory '{dir_path}': {e}")
        else:
            raise DirectoryError(f"Directory creation aborted by user as '{dir_path}' already exists.")
    
    try:
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created successfully.")
    except Exception as e:
        raise DirectoryError(f"Failed to create directory '{dir_path}': {e}")

def copy_template_with_substitution(template_file_path: Union[str, pathlib.Path], 
                                    output_file_path: Union[str, pathlib.Path], 
                                    substitutions: Dict[str, Any]) -> None:
    """
    Reads a template file, performs string substitutions using `string.Template`,
    and writes the result to an output file.

    This is used for generating configuration files like .par files or .sbatch scripts
    from predefined templates by replacing placeholders (e.g., `${jobname}`)
    with actual values.

    Args:
        template_file_path (Union[str, pathlib.Path]): Path to the input template file.
        output_file_path (Union[str, pathlib.Path]): Path where the rendered output file
                                                    will be written.
        substitutions (Dict[str, Any]): A dictionary where keys are placeholder names
                                        (without the `${}` or `$`) in the template, and
                                        values are their corresponding replacements.

    Raises:
        TemplateError: If the template file is not found, if there's an I/O error during
                       file operations, if a key required by the template is missing in
                       `substitutions`, or for any other unexpected error during template processing.
    """
    try:
        with open(template_file_path, "r") as f_template:
            template_content = f_template.read()
        
        template = Template(template_content)
        rendered_content = template.substitute(substitutions)
        
        with open(output_file_path, 'w') as f_output:
            f_output.write(rendered_content)
        print(f"Successfully wrote rendered template to '{output_file_path}'")
    except FileNotFoundError as e:
        raise TemplateError(f"Template file not found: {template_file_path}. Error: {e}")
    except IOError as e:
        raise TemplateError(f"IO error processing template '{template_file_path}' to '{output_file_path}': {e}")
    except KeyError as e:
        raise TemplateError(f"Missing substitution key {e} for template '{template_file_path}'. Available keys: {list(substitutions.keys())}")
    except Exception as e:
        raise TemplateError(f"An unexpected error occurred during template processing for '{template_file_path}': {e}")

def get_default_simulation_parameters(sim_preset_name: str = DEFAULT_SIMULATION_NAME) -> Dict[str, Any]:
    """
    Constructs a dictionary of default simulation parameters, optionally applying a preset.

    This function defines the baseline parameters for a simulation. If `sim_preset_name`
    corresponds to a defined preset in `config.SIMULATION_PRESETS`, those preset values
    will override the baseline defaults. The returned dictionary is structured to be
    compatible with `pkdpipe.cli.parsecommandline`.

    Args:
        sim_preset_name (str): The name of the simulation preset to apply.
                               Defaults to `DEFAULT_SIMULATION_NAME`. If the preset is not
                               found, a warning is issued, and base defaults (or the
                               `DEFAULT_SIMULATION_NAME` preset if it exists) are used.

    Returns:
        Dict[str, Any]: A dictionary where keys are parameter names (e.g., 'nGrid', 'dBoxSize')
                        and values are dictionaries containing 'val' (default value),
                        'type' (inferred type of the value), and 'desc' (a description).
    """
    base_params = {
        'sbatch': False,
        'interact': False,
        'nodes': DEFAULT_NODES,
        'cpupert': DEFAULT_CPUS_PER_TASK,
        'gpupern': DEFAULT_GPUS_PER_NODE,
        'rundir': DEFAULT_RUN_DIR_BASE,
        'scrdir': DEFAULT_SCRATCH_DIR_BASE,
        'jobname_template': DEFAULT_JOB_NAME_TEMPLATE,
        'email': PKDGRAVEMAIL,
        'tlimit': DEFAULT_TIME_LIMIT,
        'simname': sim_preset_name,
        'cosmo': DEFAULT_COSMOLOGY_NAME,
        'scratch': False,
        'nGrid': DEFAULT_NGRID,
        'dBoxSize': DEFAULT_BOXSIZE_MPC_H,
        'dRedFrom': DEFAULT_REDSHIFT_FROM,
        'iLPT': DEFAULT_LPT_ORDER,
        'dRedTo': DEFAULT_REDSHIFT_TARGETS_STR,
        'nSteps': DEFAULT_NSTEPS_STR,
        'iOutInterval': DEFAULT_OUTPUT_INTERVAL,
    }

    if sim_preset_name in SIMULATION_PRESETS:
        preset_overrides = SIMULATION_PRESETS[sim_preset_name]
        for key, value in preset_overrides.items():
            if key in base_params:
                base_params[key] = value
            else:
                print(f"Warning: Preset '{sim_preset_name}' contains key '{key}' not in base_params.")
    elif sim_preset_name != DEFAULT_SIMULATION_NAME:
        print(f"Warning: Simulation preset '{sim_preset_name}' not found. Using defaults for '{DEFAULT_SIMULATION_NAME}'.")
        if DEFAULT_SIMULATION_NAME in SIMULATION_PRESETS:
             base_params.update(SIMULATION_PRESETS[DEFAULT_SIMULATION_NAME])

    cli_structured_params = {}
    for param_name, default_value in base_params.items():
        param_type = type(default_value)
        if param_type == str and default_value.isdigit():
            param_type = int
        elif param_type == str and default_value.replace('.', '', 1).isdigit():
             param_type = float
        elif isinstance(default_value, bool):
            param_type = bool
        
        cli_structured_params[param_name] = {
            'val': default_value,
            'type': param_type,
            'desc': f"{param_name} (default: {default_value})"
        }
    return cli_structured_params

class Simulation:
    """
    Manages the setup, configuration, and creation of pkdgrav3 simulation runs.

    This class encapsulates all the logic for preparing a simulation. It handles:
    - Parameter parsing (from defaults, presets, direct input, or CLI).
    - Directory creation and management (run and scratch directories).
    - Generation of pkdgrav3 parameter files (.par).
    - Generation of transfer function files (using the `Cosmology` class).
    - Generation of SLURM submission scripts (.sbatch).
    - Copying necessary helper scripts (e.g., `run.sh`).
    - Optionally submitting the job to SLURM or providing interactive run instructions.

    Attributes:
        params (Dict[str, Any]): A dictionary holding all parameters for the simulation.
                                 This includes paths, cosmological settings, job settings,
                                 and pkdgrav3 specific parameters. It's populated during
                                 initialization and augmented by various setup methods.
                                 A key 'jobname_actual' is added which resolves the job name
                                 from 'jobname_template'.
    """
    def __init__(self, params: Dict[str, Any] | None = None, parse_cli_args: bool = False, **kwargs):
        """
        Initializes the Simulation instance, resolving parameters from various sources.

        The order of parameter precedence is generally:
        1. Command-line arguments (if `parse_cli_args` is True).
        2. `params` dictionary provided directly.
        3. `kwargs` provided directly.
        4. Values from a simulation preset specified by `simname` (via `get_default_simulation_parameters`).
        5. Hardcoded defaults (via `get_default_simulation_parameters`).

        After parameter resolution, `_resolve_job_name` is called to set `self.params['jobname_actual']`.

        Args:
            params (Dict[str, Any] | None): A dictionary of parameters to use for the simulation.
                                            If `parse_cli_args` is True, these are used as defaults
                                            for the CLI parser. Defaults to None.
            parse_cli_args (bool): If True, parameters are parsed from command-line arguments
                                   using `pkdpipe.cli.parsecommandline`. The defaults for the
                                   parser are obtained from `get_default_simulation_parameters`.
                                   Defaults to False.
            **kwargs (Any): Additional keyword arguments that will override parameters obtained
                            from defaults or presets if `params` is None and `parse_cli_args` is False.
                            A 'simname' kwarg can be used to specify a preset.
        """
        if parse_cli_args:
            from .cli import parsecommandline
            default_params_for_cli = get_default_simulation_parameters(DEFAULT_SIMULATION_NAME)
            self.params = parsecommandline(default_params_for_cli, description='Create and manage pkdgrav3 simulations.')
        elif params is not None:
            self.params = params
        else:
            default_params_values = {k: v['val'] for k, v in get_default_simulation_parameters(kwargs.get('simname', DEFAULT_SIMULATION_NAME)).items()}
            default_params_values.update(kwargs)
            self.params = default_params_values
        
        required_keys = ['rundir', 'jobname_template', 'nGrid', 'dBoxSize', 'gpupern', 'simname', 'cosmo']
        if not all(key in self.params for key in required_keys):
            print("Warning: Initial parameters seem incomplete. Re-fetching defaults.")
            fresh_defaults = {k: v['val'] for k, v in get_default_simulation_parameters(self.params.get('simname', DEFAULT_SIMULATION_NAME)).items()}
            for req_key in required_keys:
                if req_key not in self.params:
                    self.params[req_key] = fresh_defaults[req_key]
            for k, v_dict in get_default_simulation_parameters(self.params.get('simname', DEFAULT_SIMULATION_NAME)).items():
                if k not in self.params:
                    self.params[k] = v_dict['val']

        self._resolve_job_name()
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validates simulation parameters and raises exceptions for invalid values.
        
        Raises:
            ValueError: If any parameter has an invalid value.
        """
        # Validate nGrid
        nGrid = self.params.get('nGrid')
        if nGrid is not None and (not isinstance(nGrid, int) or nGrid <= 0):
            raise ValueError(f"nGrid must be a positive integer, got: {nGrid}")
            
        # Validate dBoxSize  
        dBoxSize = self.params.get('dBoxSize')
        if dBoxSize is not None and (not isinstance(dBoxSize, (int, float)) or dBoxSize <= 0):
            raise ValueError(f"dBoxSize must be a positive number, got: {dBoxSize}")
            
        # Validate nodes
        nodes = self.params.get('nodes')
        if nodes is not None and (not isinstance(nodes, int) or nodes <= 0):
            raise ValueError(f"nodes must be a positive integer, got: {nodes}")
            
        # Validate gpupern
        gpupern = self.params.get('gpupern') 
        if gpupern is not None and (not isinstance(gpupern, int) or gpupern <= 0):
            raise ValueError(f"gpupern must be a positive integer, got: {gpupern}")

    def _resolve_job_name(self) -> None:
        """
        Resolves the actual job name from the `jobname_template` and current parameters.

        The job name template (e.g., "N{nGrid}-L{dBoxSize}-{gpupern}gpus") is populated
        using values from `self.params` like 'nGrid', 'dBoxSize', and 'gpupern'.
        If the jobname_template contains no placeholders (no '{' characters), it's used as-is.
        The resolved name is stored in `self.params['jobname_actual']`.
        If any required keys for the template are missing or if an error occurs during
        substitution, a warning is printed, and a fallback job name is generated.
        """
        template_str = self.params.get('jobname_template', DEFAULT_JOB_NAME_TEMPLATE)
        
        # If the template contains no placeholders, use it as-is (e.g., for campaign variant names)
        if '{' not in template_str:
            self.params['jobname_actual'] = template_str
            return
            
        try:
            template_keys = {
                'nGrid': self.params.get('nGrid'),
                'dBoxSize': self.params.get('dBoxSize'),
                'gpupern': self.params.get('gpupern')
            }
            
            # Check if all required keys are available and not None
            if not all(template_keys[k] is not None for k in ['nGrid', 'dBoxSize', 'gpupern']):
                print(f"Warning: Not all keys for jobname template '{template_str}' are available. Using fallback name.")
                self.params['jobname_actual'] = f"pkdpipe_run_{self.params.get('simname', 'default')}"
            else:
                # Use string format() method instead of Template class
                self.params['jobname_actual'] = template_str.format(**template_keys)

        except KeyError as e:
            print(f"Warning: Missing key {e} for job name template '{template_str}'. Using fallback name.")
            self.params['jobname_actual'] = f"pkdpipe_run_{self.params.get('simname', 'default')}"
        except (TypeError, ValueError) as e:
            print(f"Warning: Error during job name templating: {e}. Using fallback name.")
            self.params['jobname_actual'] = f"pkdpipe_run_{self.params.get('simname', 'default')}"

    def _setup_directories_and_paths(self) -> Dict[str, pathlib.Path]:
        """
        Creates the main job directory and, if specified, a scratch directory.
        Defines and returns a dictionary of key file and directory paths for the simulation.

        The job directory is created under `self.params['rundir'] / self.params['jobname_actual']`.
        If `self.params['scratch']` is True, a corresponding directory is created under
        `self.params['scrdir']`, and a symlink named "output" is created in the
        job directory pointing to this scratch space. The effective output path for pkdgrav3
        (`ach_out_name_effective`) is set accordingly.

        Uses `safemkdir` for directory creation.

        Returns:
            Dict[str, pathlib.Path]: A dictionary mapping descriptive keys (e.g., "job_dir",
                                     "parfile", "transferfile", "ach_out_name_effective")
                                     to their `pathlib.Path` objects.

        Raises:
            DirectoryError: If `safemkdir` fails to create any of the required directories.
        """
        actual_job_name = self.params.get('jobname_actual', f"pkdpipe_fallback_jobname_{time.time():.0f}")
        base_run_dir = pathlib.Path(self.params['rundir'])
        job_dir = base_run_dir / actual_job_name

        paths = {
            "job_dir": job_dir,
            "parfile": job_dir / f"{actual_job_name}.par",
            "slurmfile": job_dir / f"{actual_job_name}.sbatch",
            "logfile": job_dir / f"{actual_job_name}.log",
            "transferfile": job_dir / f"{actual_job_name}.transfer",
            "runscript_dest": job_dir / SCRIPT_RUN_SH.name,
        }

        safemkdir(str(job_dir))

        if self.params.get('scratch', False):
            base_scratch_dir = pathlib.Path(self.params['scrdir'])
            job_scr_dir = base_scratch_dir / actual_job_name
            safemkdir(str(job_scr_dir))
            paths["ach_out_name_effective"] = str(job_scr_dir)
            # Create output subdirectory in scratch
            output_dir = job_scr_dir / "output"
            safemkdir(str(output_dir))
            link_path = job_dir / "output"
            if not link_path.exists():
                 os.symlink(output_dir, link_path, target_is_directory=True)
            print(f"Using scratch directory: {job_scr_dir}. Symlinked at {link_path}")
        else:
            paths["ach_out_name_effective"] = str(job_dir)
            # Create output subdirectory in job directory
            output_dir = job_dir / "output"
            safemkdir(str(output_dir))

        return paths

    def _generate_cosmology_transfer_file(self, all_params: Dict[str, Any], transfer_file_path: pathlib.Path) -> None:
        """
        Generates a cosmology transfer function file by creating a `Cosmology` instance.

        This method extracts only cosmological parameters from the input, creates a 
        `Cosmology` object using only those parameters, computes the transfer function, 
        and writes it to the specified file path. It also extracts key effective 
        cosmological parameters from the `Cosmology` instance and stores them in 
        `self.params` with 'effective_' prefixes for later use in the pkdgrav3 .par 
        file generation.

        Args:
            all_params (Dict[str, Any]): A dictionary containing all simulation parameters.
                                        Only cosmological parameters will be extracted and
                                        passed to the Cosmology constructor.
            transfer_file_path (pathlib.Path): The path where the transfer function file
                                               should be written.
        """
        # Extract only cosmological parameters to avoid cluttering warning messages
        cosmo_params = extract_cosmological_parameters(all_params)
        
        cosmo_instance = Cosmology(
            cosmology=cosmo_params.get('cosmo', DEFAULT_COSMOLOGY_NAME), 
            **cosmo_params
        )
        cosmo_instance.writetransfer(str(transfer_file_path))
        print(f"Cosmology transfer file written to '{transfer_file_path}'")
        
        self.params['effective_h'] = cosmo_instance.params.get('h')
        self.params['effective_omegam'] = cosmo_instance.params.get('omegam_calculated')
        self.params['effective_omegal'] = cosmo_instance.params.get('omegal_calculated')
        self.params['effective_sigma8'] = cosmo_instance.params.get('sigma8_final')
        self.params['effective_ns'] = cosmo_instance.params.get('ns')
        self.params['effective_w0'] = cosmo_instance.params.get('w0', -1.0)  # Default to LCDM value
        self.params['effective_wa'] = cosmo_instance.params.get('wa', 0.0)   # Default to LCDM value

    def _generate_parameter_file(self, paths: Dict[str, pathlib.Path]) -> None:
        """
        Generates the pkdgrav3 .par (parameter) file using a template.

        This method populates the `TEMPLATE_LIGHTCONE_PAR` template with values from
        `self.params`. It performs several calculations and lookups:
        - Determines `dRedshiftLCP` (redshift for lightcone particle placement) by converting
          the box size (in Mpc) to a comoving distance and then to redshift using the
          `Cosmology` object's `chi2z`.
        - Uses 'effective' cosmological parameters (h, omegam, omegal, sigma8, ns) that were
          set by `_generate_cosmology_transfer_file` to ensure consistency between the
          transfer function and the .par file.
        - Substitutes these and other simulation parameters (like nGrid, dBoxSize, output paths)
          into the template.
        The resulting file is written to `paths["parfile"]`.

        Args:
            paths (Dict[str, pathlib.Path]): A dictionary of paths, expected to contain
                                             "transferfile" (path to the generated transfer file)
                                             and "parfile" (path where the .par file will be written),
                                             and "ach_out_name_effective" (output directory for pkdgrav3).

        Raises:
            ValueError: If `self.params['effective_h']` is missing, None, or zero, as it's critical
                        for LCP calculations.
            TemplateError: If `copy_template_with_substitution` fails.
        """
        h_val = self.params.get('effective_h')
        if h_val is None or h_val == 0:
            raise ValueError(
                f"Critical parameter 'effective_h' is missing, None, or zero ({h_val}). "
                "Cannot proceed with LCP calculations for .par file generation. "
                "Please check cosmology setup."
            )

        temp_cosmo_for_chi = Cosmology(cosmology=self.params.get('cosmo', DEFAULT_COSMOLOGY_NAME), h=h_val)
        box_size_mpc_h = float(self.params['dBoxSize'])
        chi_for_lcp_mpc = box_size_mpc_h / h_val
        dRedshiftLCP_val = temp_cosmo_for_chi.chi2z(chi_for_lcp_mpc*0.98)

        effective_ns = self.params.get('effective_ns')
        effective_h_par = self.params.get('effective_h')
        effective_omegam = self.params.get('effective_omegam')
        effective_omegal = self.params.get('effective_omegal')
        effective_sigma8 = self.params.get('effective_sigma8')
        effective_w0 = self.params.get('effective_w0')
        effective_wa = self.params.get('effective_wa')
        
        # For wCDM cosmology: if w0 or wa are non-zero, use dOmegaDE instead of dLambda
        is_wcdm = (effective_w0 is not None and effective_w0 != -1.0) or (effective_wa is not None and effective_wa != 0.0)
        if is_wcdm:
            # Set dOmegaDE = dLambda and dLambda = 0 for wCDM
            dLambda_val = 0.0
            dOmegaDE_val = effective_omegal if effective_omegal is not None else 0.7
        else:
            # Standard LCDM: use dLambda, set dOmegaDE = 0
            dLambda_val = effective_omegal if effective_omegal is not None else 0.7
            dOmegaDE_val = 0.0

        # Create the output subdirectory path with the job name
        actual_job_name = self.params.get('jobname_actual', 'pkdpipe_job')
        output_path = f"{paths['ach_out_name_effective']}/output/{actual_job_name}"
        
        par_substitutions = {
            "achOutName": output_path,
            "achTfFile": str(paths["transferfile"]),
            "dRedshiftLCP": f"{dRedshiftLCP_val:.2f}",
            "dSpectral": f"{effective_ns if effective_ns is not None else 0.96:0.6f}",
            "h": f"{effective_h_par if effective_h_par is not None else 0.7:0.6f}",
            "dOmega0": f"{effective_omegam if effective_omegam is not None else 0.3:0.6f}",
            "dLambda": f"{dLambda_val:0.6f}",
            "dOmegaDE": f"{dOmegaDE_val:0.6f}",
            "dSigma8": f"{effective_sigma8 if effective_sigma8 is not None else 0.8:0.6f}",
            "w0": f"{effective_w0 if effective_w0 is not None else -1.0:0.6f}",
            "wa": f"{effective_wa if effective_wa is not None else 0.0:0.6f}",
            "dRedFrom": f"{self.params['dRedFrom']}",
            "nGrid": f"{self.params['nGrid']:<4}",
            "dBoxSize": f"{self.params['dBoxSize']:<4}",
            "iLPT": f"{self.params['iLPT']}",
            "dRedTo": self.params['dRedTo'],
            "nSteps": self.params['nSteps'],
            "iOutInterval": f"{self.params['iOutInterval']}"
        }
        copy_template_with_substitution(TEMPLATE_LIGHTCONE_PAR, paths["parfile"], par_substitutions)

    def _generate_slurm_script(self, paths: Dict[str, pathlib.Path]) -> None:
        """
        Generates the SLURM batch submission script (.sbatch) using a template.

        This method populates the `TEMPLATE_SLURM_SH` template with job-specific parameters
        from `self.params`, such as time limit, email, node/CPU/GPU counts, job name,
        and the command to run the simulation.
        The command to run the simulation is constructed using the paths to the `run.sh` script,
        the .par file, and the log file.
        The resulting script is written to `paths["slurmfile"]`.

        Args:
            paths (Dict[str, pathlib.Path]): A dictionary of paths, expected to contain
                                             "runscript_dest" (path to the copied run.sh),
                                             "parfile" (path to the .par file),
                                             "logfile" (path for the simulation log),
                                             and "slurmfile" (path where the .sbatch script will be written).

        Raises:
            TemplateError: If `copy_template_with_substitution` fails.
        """
        run_command = f'{paths["runscript_dest"]} {paths["parfile"]} {paths["logfile"]}'
        slurm_substitutions = {
            "tlimit": self.params['tlimit'],
            "email": self.params['email'],
            "nodes": self.params['nodes'],
            "cpupert": self.params['cpupert'],
            "gpupern": self.params['gpupern'],
            "jobname": self.params.get('jobname_actual', 'pkdpipe_job'),
            "runcmd": run_command
        }
        copy_template_with_substitution(TEMPLATE_SLURM_SH, paths["slurmfile"], slurm_substitutions)

    def _copy_run_script(self, paths: Dict[str, pathlib.Path]) -> None:
        """
        Copies the master `run.sh` script (defined by `SCRIPT_RUN_SH`) to the job directory
        and makes it executable.

        The script is copied to `paths["runscript_dest"]`.

        Args:
            paths (Dict[str, pathlib.Path]): A dictionary of paths, expected to contain
                                             "runscript_dest" where the script should be copied.

        Raises:
            FileNotFoundError: If the source `SCRIPT_RUN_SH` cannot be found or if copying/
                               permission setting fails.
        """
        try:
            shutil.copy(SCRIPT_RUN_SH, paths["runscript_dest"])
            os.chmod(paths["runscript_dest"], 0o755)
            print(f"Run script copied to '{paths['runscript_dest']}' and made executable.")
        except Exception as e:
            raise FileNotFoundError(f"Failed to copy run script from '{SCRIPT_RUN_SH}' to '{paths['runscript_dest']}': {e}")

    def _handle_job_submission(self, paths: Dict[str, pathlib.Path]) -> Optional[str]:
        """
        Handles the job submission process based on `self.params['sbatch']` and
        `self.params['interact']`.

        - If `sbatch` is True, it attempts to submit the generated SLURM script
          (`paths["slurmfile"]`) using the `sbatch` command.
        - If `interact` is True (and `sbatch` is False), it prints instructions for
          running the simulation interactively.
        - If neither is True, it prints instructions for manual submission or interactive run.

        Args:
            paths (Dict[str, pathlib.Path]): A dictionary of paths, needed for constructing
                                             run commands and identifying the SLURM script.
                                             Expected keys: "slurmfile", "job_dir", "runscript_dest",
                                             "parfile", "logfile".

        Returns:
            Optional[str]: The job ID if submitted via sbatch, None otherwise.

        Raises:
            JobSubmissionError: If `sbatch` is requested but the `sbatch` command fails or is not found.
        """
        run_command_display = f'{paths["runscript_dest"]} {paths["parfile"]} {paths["logfile"]}'
        if self.params.get('sbatch', False):
            try:
                result = subprocess.run(["sbatch", str(paths["slurmfile"])], 
                                       check=True, 
                                       cwd=paths["job_dir"], 
                                       capture_output=True, 
                                       text=True)
                # sbatch output format: "Submitted batch job <job_id>"
                output = result.stdout.strip()
                if output.startswith("Submitted batch job "):
                    job_id = output.split()[-1]
                    print(f"Job submitted via sbatch: {paths['slurmfile']} (Job ID: {job_id})")
                    return job_id
                else:
                    print(f"Job submitted via sbatch: {paths['slurmfile']} (Unknown job ID format)")
                    return None
            except subprocess.CalledProcessError as e:
                raise JobSubmissionError(f"sbatch submission failed for '{paths['slurmfile']}': {e}")
            except FileNotFoundError:
                raise JobSubmissionError("sbatch command not found. Ensure SLURM is installed and in PATH.")
        elif self.params.get('interact', False):
            print("Interactive mode: Job files created.")
            print(f"To run interactively, navigate to '{paths['job_dir']}' and execute:")
            print(f"  {run_command_display}")
            return None
        else:
            print("Simulation files created. No submission requested.")
            print(f"To submit manually: sbatch {paths['slurmfile']}")
            print(f"Or run interactively in '{paths['job_dir']}': {run_command_display}")
            return None

    def create(self) -> Optional[str]:
        """
        Orchestrates the entire process of creating a pkdgrav3 simulation setup.

        This is the main public method to be called on a `Simulation` instance.
        It performs the following steps in order:
        1. Sets up directories and paths (`_setup_directories_and_paths`).
        2. Generates the cosmology transfer function file (`_generate_cosmology_transfer_file`).
        3. Generates the pkdgrav3 parameter file (`_generate_parameter_file`).
        4. Generates the SLURM submission script (`_generate_slurm_script`).
        5. Copies the `run.sh` script (`_copy_run_script`).
        6. Handles job submission or prints instructions (`_handle_job_submission`).

        Returns:
            Optional[str]: The job ID if submitted via sbatch, None otherwise.

        Prints status messages throughout the process and a final completion message.
        Catches and prints specific errors (`DirectoryError`, `TemplateError`, `JobSubmissionError`)
        as well as any other unexpected exceptions.
        """
        try:
            paths = self._setup_directories_and_paths()
            
            self._generate_cosmology_transfer_file(self.params, paths["transferfile"])
            
            self._generate_parameter_file(paths)
            self._generate_slurm_script(paths)
            self._copy_run_script(paths)
            
            job_id = self._handle_job_submission(paths)
            
            print(f"\nSimulation '{self.params.get('jobname_actual')}' setup complete in '{paths['job_dir']}'.")
            return job_id

        except (DirectoryError, TemplateError, JobSubmissionError) as e:
            print(f"Error during simulation creation: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during simulation creation: {e}")
            return None
