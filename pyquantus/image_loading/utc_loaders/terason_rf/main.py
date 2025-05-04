from pathlib import Path

from ....data_objs.image import UltrasoundRfImage
from .parser import terasonRfParser

class EntryClass(UltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    extensions = [".mat"]
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__(scan_path, phantom_path)
        
        # Supported file extensions for this loader
        assert Path(scan_path).suffix in self.extensions, f"File must end with {self.extensions}"
        
        imgData, imgInfo, refData, refInfo = terasonRfParser(scan_path, phantom_path)
        self.rf_data = imgData.rf
        self.phantom_rf_data = refData.rf
        self.bmode = imgData.bMode
        self.axial_res = imgInfo.axialRes
        self.lateral_res = imgInfo.lateralRes
