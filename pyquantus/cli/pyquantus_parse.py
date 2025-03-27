import pickle
import argparse
import importlib.metadata

DESCRIPTION = """
PyQuantUS | RF/IQ Parsing
"""

def parser_args(parser):
    parser.add_argument('parser', type=str,
                        help='Parser to use. (Options: "canon_iq", "clarius_rf", "siemens_rf", "terason_rf")')
    parser.add_argument('--output_path', type=str, default='parsed_data.pkl', help='Path to output parser results')

def signal_args(parser):
    parser.add_argument('scan_path', type=str, help='Path to scan signals')
    parser.add_argument('phantom_path', type=str, help='Path to phantom signals')
    parser.add_argument('--frame', type=int, help='Scan frame number')

def roi_args(parser):
    parser.add_argument('roi_path', type=str, help='Path to ROI signals')
    parser.add_argument('--roi_loader', type=str, default='pkl_roi',
                        help='ROI loader to use. (Options: "pkl_roi", "pkl_roi_assert_scan", "pkl_roi_assert_scan_phantom")')
    
def analysis_args(parser):
    parser.add_argument('analysis_path', type=str, help='Path to analysis config')

def main() -> int:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser_args(parser)
    signal_args(parser)
    roi_args(parser)
    analysis_args(parser)
    args = parser.parse_args()
    
    parsers = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_parse.parsers')
    }
    roi_loaders = {
        entry_point.name: entry_point
        for entry_point in importlib.metadata.entry_points().select(group='pyquantus_parse.roi_loaders')
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
        
    # Load signal data
    imgDataStruct, imgInfoStruct, refDataStruct, refInfoStruct = parser(args.scan_path, args.phantom_path)
    splineX, splineY = roi_loader(args.roi_path, args.scan_path, args.phantom_path) # Load ROI data
    
    
    
    return 0

if __name__ == '__main__':
    exit(main())