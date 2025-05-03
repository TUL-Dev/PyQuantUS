import importlib
from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def config_loader_args(parser: ArgumentParser):
    parser.add_argument('config_path', type=str, help='Path to analysis config')
    parser.add_argument('--config_loader', type=str, default='pkl_utc',
                        help='Analysis config loader to use. See "pyquantus_parse.analysis_config_loaders" in pyproject.toml for available analysis config loaders.')
    parser.add_argument('--config_kwargs', type=str, default='{}',
                        help='Analysis config kwargs in JSON format needed for analysis class.')
    
    
def get_config_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    
    functions = {name: obj for name, obj in globals().items() if callable(obj) and obj.__module__ == 'pyquantus.analysis_config.utc_config.functions'}
    return functions
