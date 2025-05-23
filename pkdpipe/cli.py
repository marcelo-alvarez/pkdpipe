"""
pkdpipe command-line interface (CLI) utilities.

This module provides functions for parsing command-line arguments
and entry points for CLI commands.
"""
import sys
import os
import importlib.util
import argparse
from typing import Dict, Any, Optional, List

from .config import (
    DEFAULT_SIMULATION_NAME,
)

def parsecommandline(param_definitions: Dict[str, Dict[str, Any]], description: str) -> Dict[str, Any]:
    """
    Parses command-line arguments based on a dictionary of parameter definitions.

    Args:
        param_definitions: A dictionary where keys are parameter names and values are
                           dictionaries containing 'val' (default value),
                           'type', and 'desc' (description).
        description: A description of the command-line program.

    Returns:
        A dictionary of parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=description)

    for param_name, details in param_definitions.items():
        default_val = details.get('val')
        param_type = details.get('type')
        param_desc = details.get('desc', f'Set {param_name}')
        required = details.get('required', False)

        arg_name = '--' + param_name

        if param_type is bool:
            if default_val is True:
                parser.add_argument(arg_name, action='store_false', help=f'{param_desc} (flag, default: True)')
            else:
                parser.add_argument(arg_name, action='store_true', help=f'{param_desc} (flag, default: False)')
        elif param_type is list or isinstance(default_val, list):
            parser.add_argument(
                arg_name, 
                default=default_val, 
                help=f'{param_desc} (space-separated list, default: {default_val})',
                type=details.get('element_type', str),
                nargs='*',
                required=required
            )
        else:
            parser.add_argument(
                arg_name, 
                default=default_val, 
                help=f'{param_desc} (default: {default_val})', 
                type=param_type,
                required=required
            )

    parsed_args = parser.parse_args()
    return vars(parsed_args)

def create() -> None:  # Renamed from create_simulation_cli
    """
    CLI entry point for creating a pkdgrav3 simulation.
    Uses Simulation class with `parse_cli_args=True`.
    """
    print("Initializing simulation creation from CLI...")
    from .simulation import Simulation
    try:
        sim_instance = Simulation(parse_cli_args=True)
        sim_instance.create()
        print("Simulation creation process finished.")
    except Exception as e:
        print(f"An error occurred in create: {e}")

if __name__ == '__main__':
    print("Running CLI directly for testing create...")
    create()
