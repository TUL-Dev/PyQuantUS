import importlib
from pathlib import Path

from argparse import ArgumentParser

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
    current_dir = Path(__file__).parent
    classes = {}
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(f"pyquantus.cli.config_loaders.{folder.name}.main")
                entry_class = getattr(module, "EntryClass", None)
                if entry_class:
                    classes[folder.name] = entry_class
            except ModuleNotFoundError:
                # Handle the case where the module cannot be found
                pass
    
    return classes
