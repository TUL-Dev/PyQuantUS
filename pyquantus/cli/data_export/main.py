import importlib
from pathlib import Path

from argparse import ArgumentParser

def data_export_args(parser: ArgumentParser):
    parser.add_argument('data_export_type', type=str, default='',
                        help='Data export type to use. Available data export types: ' + ', '.join(get_data_export_types().keys()))
    parser.add_argument('data_export_path', type=str,
                        help='Path to save exported numerical data to. Must end in .csv or .pkl')
    parser.add_argument('--data_export_kwargs', type=str, default='{}',
                        help='Data export kwargs in JSON format needed for data export class.')
    
    
def get_data_export_types() -> dict:
    """Get visualization types for the CLI.
    
    Returns:
        dict: Dictionary of visualization types.
    """
    current_dir = Path(__file__).parent
    classes = {}
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(f"pyquantus.cli.data_export.{folder.name}.main")
                entry_class = getattr(module, "EntryClass", None)
                if entry_class:
                    classes[folder.name] = entry_class
            except ModuleNotFoundError:
                # Handle the case where the module cannot be found
                pass
    
    return classes
