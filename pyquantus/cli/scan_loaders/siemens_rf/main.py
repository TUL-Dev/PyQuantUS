from .._obj.us_rf import UltrasoundRfImage
from .parser import siemensRfParser

class EntryClass(UltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__(scan_path, phantom_path)
        
        frame = kwargs.get("frame", None)
        if frame is None:
            raise ValueError("Frame must be provided for Clarius RF data.")
        
        # Load signal data
        _, imgData, imgInfo, refData, refInfo = siemensRfParser(scan_path, phantom_path)
        
        # Package data
        self.bmode = imgData.bMode[frame]
        self.rf_data = imgData.rf[frame]
        self.phantom_rf_data = refData.rf[frame]
        self.axial_res = imgInfo.depth / imgData.rf[frame].shape[0]
        self.lateral_res = self.axial_res * (
            imgData.rf[frame].shape[0]/imgData.rf[frame].shape[1]
        ) # placeholder
