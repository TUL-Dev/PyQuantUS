from .._obj.us_rf import UltrasoundRfImage
from .parser import terasonRfParser

class EntryClass(UltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__()
        
        imgData, imgInfo, refData, refInfo = terasonRfParser(scan_path, phantom_path)
        self.rf_data = imgData.rf
        self.phantom_rf_data = refData.rf
        self.bmode = imgData.bMode
        self.axial_res = imgInfo.axialRes
        self.lateral_res = imgInfo.lateralRes
