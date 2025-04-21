import importlib
from pathlib import Path

from argparse import ArgumentParser

def visualization_args(parser: ArgumentParser):
    parser.add_argument('visualization_type', type=str, default='paramap_drawing',
                        help='Visualization type to use. Available visualization types: ' + ', '.join(get_visualization_types().keys()))
    parser.add_argument('--visualization_kwargs', type=str, default='{}',
                        help='Visualization kwargs in JSON format needed for visualization class.')
    parser.add_argument('--visualization_output_path', type=str, default='visualizations.pkl',
                        help='Path to output visualization class instance')
    parser.add_argument('--save_visualization_class', type=bool, default=False,
                        help='Save visualization class instance to VISUALIZATION_OUTPUT_PATH')
    
    
def get_visualization_types() -> dict:
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
                module = importlib.import_module(f"pyquantus.cli.visualizations.{folder.name}.main")
                entry_class = getattr(module, "EntryClass", None)
                if entry_class:
                    classes[folder.name] = entry_class
            except ModuleNotFoundError:
                # Handle the case where the module cannot be found
                pass
    
    return classes
