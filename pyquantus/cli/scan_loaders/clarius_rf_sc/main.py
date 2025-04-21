from pathlib import Path
from typing import Tuple

from .._obj.us_rf import ScUltrasoundRfImage
from .parser import ClariusTarUnpacker, clariusRfParser

class EntryClass(ScUltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__()
        
        frame = kwargs.get("frame", None)
        if frame is None:
            raise ValueError("Frame must be provided for Clarius RF data.")
        
        def find_info_paths(path: str) -> Tuple[str, str, str]:
            """Find the paths to the RF, TGC, and info files."""
            if path.endswith(".tar"):
                ClariusTarUnpacker(path, "single_tar")
                unpackedTarFolder = Path(path.replace(".tar", "_extracted"))
                rf_path = ""; tgc_path = ""; info_path = ""
                for file in unpackedTarFolder.iterdir():
                    if file.name.endswith("_rf.raw"):
                        rf_path = str(file)
                    elif file.name.endswith("_env.tgc.yml"):
                        tgc_path = str(file)
                    elif file.name.endswith("_rf.yml"):
                        info_path = str(file)
                if rf_path == "" or info_path == "":
                    raise Exception("Missing files in tar")
            elif path.endswith(".raw"):
                rf_path = path.replace(".raw", "_rf.raw")
                tgc_path = path.replace("_rf.raw", "_env.tgc.yml")
                info_path = path.replace(".raw", ".yml")
            else:
                raise ValueError("Unsupported file format")
            if not len(tgc_path) or not Path(tgc_path).exists():
                tgc_path = None
            return rf_path, tgc_path, info_path
        
        rf_path, tgc_path, info_path = find_info_paths(scan_path)
        ref_rf_path, ref_tgc_path, ref_info_path = find_info_paths(phantom_path)
            
        img_data, img_info, ref_data, ref_info, sc = clariusRfParser(rf_path, tgc_path, info_path, 
                                                                 ref_rf_path, ref_tgc_path, ref_info_path)
        
        assert sc, "Data must be scan converted for this loader."
        self.rf_data = img_data.rf[frame]
        self.phantom_rf_data = ref_data.rf[frame]
        self.bmode = img_data.bMode[frame]
        self.axial_res = img_info.depth / self.rf_data.shape[0]
        self.lateral_res = self.axial_res * (
            self.rf_data.shape[0] / self.rf_data.shape[1]
        ) # placeholder
        self.sc_axial_res = img_info.axialRes
        self.sc_lateral_res = img_info.lateralRes
        self.sc_bmode = img_data.scBmode[frame]
        self.xmap = img_data.scBmodeStruct.xmap
        self.ymap = img_data.scBmodeStruct.ymap
        self.tilt = img_info.tilt1
        self.width = img_info.width1
        self.start_depth = img_info.startDepth1
        self.end_depth = img_info.endDepth1
