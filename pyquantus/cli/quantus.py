import json
import yaml
import pickle
import argparse
import importlib.metadata
from pathlib import Path

DESCRIPTION = """
QuantUS | RF/IQ Parsing -> UTC Analysis -> Visualizations
"""

def parser_args(parser):
    parser.add_argument('parser', type=str,
                        help='Parser to use. See "pyquantus_parse.parsers" in pyproject.toml for available parsers.')
    parser.add_argument('--parser_output_path', type=str, default='parsed_data.pkl', help='Path to output parser results')
    parser.add_argument('--save_parsed_results', type=bool, default=False, 
                        help='Save parsed results to PARSER_OUTPUT_PATH')

def signal_args(parser):
    parser.add_argument('scan_path', type=str, help='Path to scan signals')
    parser.add_argument('phantom_path', type=str, help='Path to phantom signals')

def roi_args(parser):
    parser.add_argument('roi_path', type=str, help='Path to ROI signals')
    parser.add_argument('--roi_loader', type=str, default='pkl_roi',
                        help='ROI loader to use. See "pyquantus_parse.roi_loaders" in pyproject.toml for available ROI loaders.')
    
def analysis_config_args(parser):
    parser.add_argument('analysis_config_path', type=str, help='Path to analysis config')
    parser.add_argument('--analysis_config_loader', type=str, default='pkl_config',
                        help='Analysis config loader to use. See "pyquantus_parse.analysis_config_loaders" in pyproject.toml for available analysis config loaders.')
    
def analysis_args(parser):
    parser.add_argument('analysis_class', type=str, default='base',
                        help='Analysis class to use. See "pyquantus_utc.analysis_classes" in pyproject.toml for available analysis classes.')
    parser.add_argument('--analysis_kwargs', type=str, default='{}',
                        help='Analysis kwargs in JSON format needed for analysis class.')
    parser.add_argument('--analysis_output_path', type=str, default='analysis_results.pkl',
                        help='Path to output analysis results')
    parser.add_argument('--save_analysis_results', type=bool, default=False,
                        help='Save analysis results to ANALYSIS_OUTPUT_PATH')
    
def visualization_args(parser):
    parser.add_argument('visualization_class', type=str, default='base',
                        help='Visualization class to use. See "pyquantus_utc.visualization_classes" in pyproject.toml for available visualization classes.')
    parser.add_argument('--visualization_computation_kwargs', type=str, default='{}',
                        help='Visualization kwargs in JSON format needed for computational aspect of visualization class.')
    parser.add_argument('--visualization_export_kwargs', type=str, default='{}',
                        help='Visualization kwargs in JSON format needed for exporting aspect of visualization class.')
    parser.add_argument('--export_visualizations', type=bool, default=False,
                        help='Whether or not to complete visualizatios export step')
    parser.add_argument('--visualization_output_path', type=str, default='visualizations.pkl',
                        help='Path to output visualization class instance')
    parser.add_argument('--save_visualization_class', type=bool, default=False,
                        help='Save visualization class instance to VISUALIZATION_OUTPUT_PATH')
    
def main_cli() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser_args(parser)
    signal_args(parser)
    roi_args(parser)
    analysis_config_args(parser)
    analysis_args(parser)
    visualization_args(parser)
    args = parser.parse_args()
    args.analysis_kwargs = json.loads(args.analysis_kwargs)
    args.visualization_computation_kwargs = json.loads(args.visualization_computation_kwargs)
    args.visualization_export_kwargs = json.loads(args.visualization_export_kwargs)
    
    return core_pipeline(args)    

    
def main_yaml() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config, **vars(args))
    args.analysis_kwargs = {} if args.analysis_kwargs is None else args.analysis_kwargs
    args.visualization_computation_kwargs = {} if args.visualization_computation_kwargs is None else args.visualization_computation_kwargs
    args.visualization_export_kwargs = {} if args.visualization_export_kwargs is None else args.visualization_export_kwargs
    
    return core_pipeline(args)
    
def core_pipeline(args) -> int:
    parsers = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_parse.parsers')
    }
    roi_loaders = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_parse.roi_loaders')
    }
    analysis_config_loaders = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_parse.analysis_config_loaders')
    }
    analysis_classes = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_utc.analysis_classes')
    }
    analysis_kwarg_types = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_utc.analysis_kwarg_types')
    }
    visualization_classes = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_visualizations.visualization_classes')
    }
    visualization_computation_kwarg_types = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_visualizations.visualization_computation_kwarg_types')
    }
    visualization_export_kwarg_types = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_visualizations.visualization_export_kwarg_types')
    }
    
    # Get applicable plugins
    try:
        parser = parsers[args.parser].load()
    except KeyError:
        print(f'Parser "{args.parser}" is not available!')
        print(f"Available parsers: {', '.join(parsers.keys())}")
        return 1
    try:
        roi_loader = roi_loaders[args.roi_loader].load()
    except KeyError:
        print(f'ROI loader "{args.roi_loader}" is not available!')
        print(f"Available ROI loaders: {', '.join(roi_loaders.keys())}")
        return 1
    try:
        analysis_config_loader = analysis_config_loaders[args.analysis_config_loader].load()
    except KeyError:
        print(f'Analysis config loader "{args.analysis_config_loader}" is not available!')
        print(f"Available analysis config loaders: {', '.join(analysis_config_loaders.keys())}")
        return 1
    try:
        analysis_class = analysis_classes[args.analysis_class].load()
    except KeyError:
        print(f'Analysis class "{args.analysis_class}" is not available!')
        print(f"Available analysis classes: {', '.join(analysis_classes.keys())}")
        return 1
    try:
        analysis_kwarg_type = analysis_kwarg_types[args.analysis_class].load()
    except KeyError:
        print(f'Analysis kwargs supporting the analysis class "{args.analysis_class}" is not available!')
        print(f"Available analysis classes with kwargs support are: {', '.join(analysis_kwarg_types.keys())}")
        return 1
    try:
        visualization_class = visualization_classes[args.visualization_class].load()
    except KeyError:
        print(f'Visualization class "{args.visualization_class}" is not available!')
        print(f"Available visualization classes: {', '.join(visualization_classes.keys())}")
        return 1
    try:
        visualization_computation_kwarg_type = visualization_computation_kwarg_types[args.visualization_class].load()
    except KeyError:
        print(f'Visualization computation kwargs supporting the visualization class "{args.visualization_class}" is not available!')
        print(f"Available visualization classes with computation kwargs support are: {', '.join(visualization_computation_kwarg_types.keys())}")
        return 1
    try:
        visualization_export_kwarg_type = visualization_export_kwarg_types[args.visualization_class].load()
    except KeyError:
        print(f'Visualization export kwargs supporting the visualization class "{args.visualization_class}" is not available!')
        print(f"Available visualization classes with export kwargs support are: {', '.join(visualization_export_kwarg_types.keys())}")
        return 1
    
    # Parsing / data loading
    spline_x, spline_y, frame = roi_loader(args.roi_path, args.scan_path, args.phantom_path) # Load ROI data
    ultrasound_image = parser(args.scan_path, args.phantom_path, frame) # Load signal data
    analysis_config = analysis_config_loader(
        args.analysis_config_path, args.scan_path, args.phantom_path) # Load analysis config
    
    if args.save_parsed_results:
        # Save parsed data
        parsed_data = [ultrasound_image, analysis_config, spline_x, spline_y]
        if Path(args.output_path).is_dir():
            output_path = Path(args.output_path) / 'parsed_data.pkl'
        elif Path(args.output_path).suffix != '.pkl':
            output_path = Path(args.output_path).with_suffix('.pkl')
        else:
            output_path = Path(args.output_path)
        with open(output_path, 'wb') as f:
            pickle.dump(parsed_data, f)
    
    # Analysis
    analysis_obj = analysis_class(analysis_kwarg_type, **args.analysis_kwargs)
    analysis_obj.ultrasoundImage = ultrasound_image
    analysis_obj.config = analysis_config
    analysis_obj.bmodeSplineX = spline_x
    analysis_obj.bmodeSplineY = spline_y
    analysis_obj.generateRoiWindows()
    analysis_obj.computeParamaps()
    analysis_obj.computeSingleWindow()
    
    if args.save_analysis_results:
        # Save analysis results
        if Path(args.analysis_output_path).is_dir():
            output_path = Path(args.analysis_output_path) / 'analysis_results.pkl'
        elif Path(args.analysis_output_path).suffix != '.pkl':
            output_path = Path(args.analysis_output_path).with_suffix('.pkl')
        else:
            output_path = Path(args.analysis_output_path)
        with open(output_path, 'wb') as f:
            pickle.dump(analysis_obj, f)
            
    # Visualizations
    visualization_obj = visualization_class(analysis_obj)
    visualization_obj.computeVisualizations(visualization_computation_kwarg_type, **args.visualization_computation_kwargs)
    
    if args.export_visualizations:
        visualization_obj.exportVisualizations(visualization_export_kwarg_type, **args.visualization_export_kwargs)
        
    if args.save_visualization_class:
        # Save visualization class instance
        if Path(args.visualization_output_path).is_dir():
            output_path = Path(args.visualization_output_path) / 'visualizations.pkl'
        elif Path(args.visualization_output_path).suffix != '.pkl':
            output_path = Path(args.visualization_output_path).with_suffix('.pkl')
        else:
            output_path = Path(args.visualization_output_path)
        with open(output_path, 'wb') as f:
            pickle.dump(visualization_obj, f)
    
    return 0

if __name__ == '__main__':
    exit(main_cli())
