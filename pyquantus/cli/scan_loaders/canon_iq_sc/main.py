from .._obj.us_rf import ScUltrasoundRfImage
from .parser import canonIqParser

class EntryClass(ScUltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__(scan_path, phantom_path)
        
        imgData, imgInfo, refData, refInfo = canonIqParser(scan_path, phantom_path)
        self.rf_data = imgData.rf
        self.phantom_rf_data = refData.rf
        self.bmode = imgData.bMode
        self.axial_res = imgInfo.depth / imgData.rf.shape[0]
        self.lateral_res = self.axial_res * (
            imgData.rf.shape[0] / imgData.rf.shape[1]
        ) # placeholder
        self.sc_axial_res = imgInfo.axialRes
        self.sc_lateral_res = imgInfo.lateralRes
        self.sc_bmode = imgData.scBmode
        self.xmap = imgData.scBmodeStruct.xmap
        self.ymap = imgData.scBmodeStruct.ymap
        self.tilt = imgInfo.tilt1
        self.width = imgInfo.width1
        self.start_depth = imgInfo.startDepth1
        self.end_depth = imgInfo.endDepth1
