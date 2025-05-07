from pyquantus.data_objs import UltrasoundRfImage
from .parser import siemensRfParser

class EntryClass(UltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    extensions = [".rfd"]
    spatial_dims = 2
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__(scan_path, phantom_path)
        
        # Load signal data
        _, imgData, imgInfo, refData, refInfo = siemensRfParser(scan_path, phantom_path)
        
        # Package data
        self.bmode = imgData.bMode
        self.rf_data = imgData.rf
        self.phantom_rf_data = refData.rf[0]
        self.axial_res = imgInfo.depth / imgData.rf.shape[1]
        self.lateral_res = self.axial_res * (
            imgData.rf.shape[1]/imgData.rf.shape[2]
        ) # placeholder
