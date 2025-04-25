from .us_rf import RfAnalysisConfig

class RfAnalysisConfig3d(RfAnalysisConfig):
    """
    Class to store configuration data for RF analysis.
    """

    def __init__(self):
        super().__init__()
        self.cor_win_size: float  # coronal width per window (mm)
        self.coronal_overlap: float # % of cor window length to move before next window
