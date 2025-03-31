import numpy as np

from pyquantus.utc.analysis import UtcAnalysis

# Specify format for analysis kwargs, which are new parameters
# needed for this analysis not already in the UtcAnalysis class.
analysisKwargs = {} # REQUIRED
        
class BaseAnalysis(UtcAnalysis):
    """Base analysis class for UTC analysis. Only computes MBF, SS, and SI.
    """
    def __init__(self, kwargTypes, **kwargs):
        super().__init__(kwargTypes, **kwargs)