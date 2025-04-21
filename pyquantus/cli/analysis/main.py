import importlib
from pathlib import Path

from argparse import ArgumentParser

def analysis_args(parser: ArgumentParser):
    parser.add_argument('analysis_type', type=str, default='spectral_paramap',
                        help='Analysis type to complete. Available analysis types: ' + ', '.join(get_analysis_types().keys()))
    parser.add_argument('--analysis_kwargs', type=str, default='{}',
                        help='Analysis kwargs in JSON format needed for analysis class.')
    parser.add_argument('--analysis_output_path', type=str, default='analysis_results.pkl',
                        help='Path to output analysis results')
    parser.add_argument('--save_analysis_results', type=bool, default=False,
                        help='Save analysis results to ANALYSIS_OUTPUT_PATH')
    
    
def get_analysis_types() -> dict:
    """Get analysis types for the CLI.
    
    Returns:
        dict: Dictionary of analysis types.
    """
    current_dir = Path(__file__).parent
    classes = {}
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(f"pyquantus.cli.analysis.{folder.name}.main")
                entry_class = getattr(module, "EntryClass", None)
                if entry_class:
                    classes[folder.name] = entry_class
            except ModuleNotFoundError:
                # Handle the case where the module cannot be found
                pass
    
    return classes
