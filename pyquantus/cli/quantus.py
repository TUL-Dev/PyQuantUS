import json
import yaml
import pickle
import argparse
import importlib.metadata
from pathlib import Path

from .scan_loaders.main import get_scan_loaders, scan_loader_args
from .seg_loaders.main import get_seg_loaders, seg_loader_args
from .config_loaders.main import get_config_loaders, config_loader_args
from .analysis.main import get_analysis_types, analysis_args
from .visualizations.main import get_visualization_types, visualization_args
from .data_export.main import get_data_export_types, data_export_args

DESCRIPTION = """
QuantUS | Custom US Analysis Workflows
"""
    
def main_cli() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    scan_loader_args(parser)
    seg_loader_args(parser)
    config_loader_args(parser)
    analysis_args(parser)
    visualization_args(parser)
    data_export_args(parser)
    args = parser.parse_args()
    args.scan_loader_kwargs = json.loads(args.scan_loader_kwargs)
    args.seg_loader_kwargs = json.loads(args.seg_loader_kwargs)
    args.config_kwargs = json.loads(args.config_kwargs)
    args.analysis_kwargs = json.loads(args.analysis_kwargs)
    args.visualization_kwargs = json.loads(args.visualization_kwargs)
    
    return core_pipeline(args)    

def main_yaml() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config, **vars(args))
    args.scan_loader_kwargs = {} if args.scan_loader_kwargs is None else args.scan_loader_kwargs
    args.seg_loader_kwargs = {} if args.seg_loader_kwargs is None else args.seg_loader_kwargs
    args.config_kwargs = {} if args.config_kwargs is None else args.config_kwargs
    args.analysis_kwargs = {} if args.analysis_kwargs is None else args.analysis_kwargs
    args.visualization_kwargs = {} if args.visualization_kwargs is None else args.visualization_kwargs
    args.data_export_kwargs = {} if args.data_export_kwargs is None else args.data_export_kwargs
    
    return core_pipeline(args)
    
def core_pipeline(args) -> int:
    scan_loaders = get_scan_loaders()
    seg_loaders = get_seg_loaders()
    config_loaders = get_config_loaders()
    analysis_types = get_analysis_types()
    visualization_types = get_visualization_types()
    data_export_types = get_data_export_types()
    
    # Get applicable plugins
    try:
        scan_loader = scan_loaders[args.scan_loader]
    except KeyError:
        print(f'Parser "{args.scan_loader}" is not available!')
        print(f"Available parsers: {', '.join(scan_loaders.keys())}")
        return 1
    try:
        seg_loader = seg_loaders[args.seg_loader]
    except KeyError:
        print(f'Segmentation loader "{args.seg_loader}" is not available!')
        print(f"Available segmentation loaders: {', '.join(seg_loaders.keys())}")
        return 1
    try:
        config_loader = config_loaders[args.config_loader]
    except KeyError:
        print(f'Analysis config loader "{args.config_loader}" is not available!')
        print(f"Available analysis config loaders: {', '.join(config_loaders.keys())}")
        return 1
    try:
        analysis_class = analysis_types[args.analysis_type]
    except KeyError:
        print(f'Analysis type "{args.analysis_type}" is not available!')
        print(f"Available analysis types: {', '.join(analysis_types.keys())}")
        return 1
    try:
        visualization_class = visualization_types[args.visualization_type]
    except KeyError:
        print(f'Visualization type "{args.visualization_type}" is not available!')
        print(f"Available visualization types: {', '.join(visualization_types.keys())}")
        return 1
    try:
        data_export_class = data_export_types[args.data_export_type]
    except KeyError:
        print(f'Data export type "{args.data_export_type}" is not available!')
        print(f"Available data export types: {', '.join(data_export_types.keys())}")
        return 1
    
    # Parsing / data loading
    seg_data = seg_loader(args.seg_path, scan_path=args.scan_path, phantom_path=args.phantom_path, **args.seg_loader_kwargs) # Load seg data
    image_data = scan_loader(args.scan_path, args.phantom_path, frame=seg_data.frame, **args.scan_loader_kwargs) # Load signal data
    config = config_loader(args.config_path, scan_path=args.scan_path, phantom_path=args.phantom_path, **args.config_kwargs) # Load analysis config
    
    # Analysis
    analysis_obj = analysis_class(image_data, config, seg_data, **args.analysis_kwargs)
    analysis_obj.compute_paramaps()
    analysis_obj.compute_single_window()
            
    # Visualizations
    visualization_obj = visualization_class(analysis_obj, **args.visualization_kwargs)
    visualization_obj.compute_visualizations()
            
    # Data Export
    data_export_obj = data_export_class(analysis_obj, args.data_export_path, **args.data_export_kwargs)
    data_export_obj.save_data()
    
    return 0

if __name__ == '__main__':
    exit(main_cli())
