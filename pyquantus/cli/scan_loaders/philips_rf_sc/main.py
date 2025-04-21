from .._obj.us_rf import ScUltrasoundRfImage
from .philipsMat import philips2dRfMatParser
from .philipsRf import philipsRfParser

class EntryClass(ScUltrasoundRfImage):
    """
    Class for Terason RF image data.
    This class is used to parse RF data from Terason ultrasound machines.
    """
    
    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        super().__init__()
        
        frame = kwargs.get("frame", 0)
        assert isinstance(frame, int)
        
        def get_mat_path(path: str):
            if path.endswith(".rf"):
                mat_path = path.replace('.rf', '.mat')
                philipsRfParser(path)
            elif path.endswith(".mat"):
                mat_path = path
            else:
                raise ValueError(f"Inputted scan {scan_path} is not a .mat file!")
            assert mat_path.endswith('.mat')
            return mat_path
        
        mat_scan_path = get_mat_path(scan_path)
        mat_ref_path = get_mat_path(phantom_path)
        img_data, img_info, ref_data, ref_info = philips2dRfMatParser(mat_scan_path, mat_ref_path, frame)
        
        self.rf_data = img_data.rf
        self.phantom_rf_data = ref_data.rf
        self.bmode = img_data.bMode
        self.sc_bmode = img_data.scBmode
        self.axial_res = img_info.depth / img_data.rf.shape[0]
        self.lateral_res = self.axial_res * (
            img_data.rf.shape[0] / img_data.rf.shape[1]
        ) # placeholder
        self.sc_axial_res = img_info.axialRes
        self.sc_lateral_res = img_info.lateralRes
        self.xmap = img_data.scBmodeStruct.xmap
        self.ymap = img_data.scBmodeStruct.ymap
        self.width = img_info.width1
        self.tilt = img_info.tilt1
        self.start_depth = img_info.startDepth1
        self.end_depth = img_info.endDepth1
