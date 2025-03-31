from typing import Any

from pyquantus.utc.transforms import checkKwargs
from pyquantus.utc.visualizations import UtcVisualizations

computationKwargs = {}  # REQUIRED
exportKwargs = {}       # REQUIRED

class BaseVisualizations(UtcVisualizations):
    """Class to facilitate and store UTC data and images after analysis.
    This class supports both scan converted and non-scan converted images. 
    This class computes and exports no visualizations by default.
    
    Attributes:
        utcAnalysis (UtcAnalysis): UTC analysis object     
    """
    def __init__(self, analysisObj: Any):
        super().__init__(analysisObj)
    
    def computeVisualizations(self, kwargTypes: dict, **kwargs):
        """Used to specify which visualizations to compute.
        """
        assert checkKwargs(kwargTypes, kwargs), "Kwargs are inconsistent with kwargTypes"
        
        pass

    def exportVisualizations(self, kwargTypes: dict, **kwargs):
        """Used to specify which visualizations to export and where.
        """
        assert checkKwargs(kwargTypes, kwargs), "Kwargs are inconsistent with kwargTypes"
        
        pass
