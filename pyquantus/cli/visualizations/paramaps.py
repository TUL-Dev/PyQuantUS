from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt

from pyquantus.utc.transforms import checkKwargs
from pyquantus.utc.visualizations import UtcVisualizations

computationKwargs = {}  # REQUIRED
exportKwargs = {
    'psPlotOutputPath': str, # path to PNG file
    'paramapFolderPath': str
    }       # REQUIRED

class ParamapVisualizations(UtcVisualizations):
    """Class to facilitate and store UTC data and images after analysis.
    This class supports both scan converted and non-scan converted images. 
    This class computes and exports all visualizations included in
    PyQuantUS's UtcVisualization class by default.
    
    Attributes:
        utcAnalysis (UtcAnalysis): UTC analysis object     
    """
    def __init__(self, analysisObj: Any):
        super().__init__(analysisObj)
        self.psPlot: plt.Figure
    
    def computeVisualizations(self, kwargTypes: dict, **kwargs):
        """Used to specify which visualizations to compute.
        """
        assert checkKwargs(kwargTypes, kwargs), "Kwargs are inconsistent with kwargTypes"
        
        self.drawParamaps()             # compute all paramatric maps
        self.psPlot = self.plotPsData() # compute power spectrum plot

    def exportVisualizations(self, kwargTypes: dict, **kwargs):
        """Used to specify which visualizations to export and where.
        """
        assert checkKwargs(kwargTypes, kwargs), "Kwargs are inconsistent with kwargTypes"
        assert kwargs['psPlotOutputPath'].endswith('.png'), "Power spectrum plot output path must end with .png"
        
        # Save the power spectrum plot
        psPlotOutputPath = kwargs['psPlotOutputPath']
        self.psPlot.savefig(psPlotOutputPath)
        
        # Save the parametric maps
        paramapFolderPath = Path(kwargs['paramapFolderPath'])
        paramapFolderPath.mkdir(parents=True, exist_ok=True)
        if hasattr(self.utcAnalysis.ultrasoundImage, 'scBmode'):
            for paramIx, param in enumerate(self.params):
                plt.imsave(paramapFolderPath / f'{param}.png', self.scParamaps[paramIx])
        else:
            for paramIx, param in enumerate(self.params):
                plt.imsave(paramapFolderPath / f'{param}.png', self.paramaps[paramIx])
