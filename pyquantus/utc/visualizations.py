from typing import List, Any

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyquantus.parse.transforms import scanConvert
from pyquantus.utc.transforms import condenseArr, expandArr

class UtcVisualizations:
    """Class to facilitate and store UTC data and images after analysis.
    This class supports both scan converted and non-scan converted images. 
    
    Attributes:
        utcAnalysis (UtcAnalysis): UTC analysis object 
        params (List[str]): List of single quantitative parameters
        minParamVals (List[float]): List of minimum values for each parameter
        maxParamVals (List[float]): List of maximum values for each parameter
        paramaps (List[np.ndarray]): List of parametric maps for each parameter
        scParamaps (List[np.ndarray]): List of scan converted parametric maps for each parameter
        windowIdxMap (np.ndarray): Map of window indices for each pixel
        scWindowIdxMap (np.ndarray): Scan converted map of window indices for each pixel
        cmaps (List[np.ndarray]): List of colormaps for parametric maps    
    """
    def __init__(self, utcAnalysis: Any):
        self.utcAnalysis = utcAnalysis
        self.rectCoords: List[int]
    
        self.params: List[str]; self.minParamVals: List[float]; self.maxParamVals: List[float]
        self.paramaps: List[np.ndarray]; self.scParamaps: List[np.ndarray]; self.legendParamaps: List[plt.Figure]
        self.windowIdxMap: np.ndarray; self.scWindowIdxMap: np.ndarray
        
        # Cmap library
        summerCmap = plt.get_cmap("summer")
        summerCmap = [summerCmap(i)[:3] for i in range(summerCmap.N)]
        winterCmap = plt.get_cmap("winter")
        winterCmap = [winterCmap(i)[:3] for i in range(winterCmap.N)]
        autunmCmap = plt.get_cmap("autumn")
        autunmCmap = [autunmCmap(i)[:3] for i in range(autunmCmap.N)]
        springCmap = plt.get_cmap("spring")
        springCmap = [springCmap(i)[:3] for i in range(springCmap.N)]
        coolCmap = plt.get_cmap("cool")
        coolCmap = [coolCmap(i)[:3] for i in range(coolCmap.N)]
        hotCmap = plt.get_cmap("hot")
        hotCmap = [hotCmap(i)[:3] for i in range(hotCmap.N)]
        boneCmap = plt.get_cmap("bone")
        boneCmap = [boneCmap(i)[:3] for i in range(boneCmap.N)]
        copperCmap = plt.get_cmap("copper")
        copperCmap = [copperCmap(i)[:3] for i in range(copperCmap.N)]
        self.cmaps = [np.array(plt.get_cmap("viridis").colors), np.array(plt.get_cmap("magma").colors),
                      np.array(plt.get_cmap("plasma").colors), np.array(plt.get_cmap("inferno").colors),
                      np.array(plt.get_cmap("cividis").colors), np.array(summerCmap),
                      np.array(winterCmap), np.array(autunmCmap), np.array(springCmap), 
                      np.array(coolCmap), np.array(hotCmap), np.array(boneCmap), np.array(copperCmap)]
        self.cmapNames = ["viridis", "magma", "plasma", "inferno", "cividis", "summer", "winter", "autumn", 
                          "spring", "cool", "hot", "bone", "copper"]

    def convertImagesToRGB(self):
        """Converts grayscale images to RGB for colormap application.
        """
        self.utcAnalysis.ultrasoundImage.bmode = cv2.cvtColor(
            np.array(self.utcAnalysis.ultrasoundImage.bmode).astype('uint8'),
            cv2.COLOR_GRAY2RGB
        )
        if hasattr(self.utcAnalysis.ultrasoundImage, 'scBmode'):
            self.utcAnalysis.ultrasoundImage.scBmode = cv2.cvtColor(
                np.array(self.utcAnalysis.ultrasoundImage.scBmode).astype('uint8'),
                cv2.COLOR_GRAY2RGB
            )

    def drawParamaps(self):
        """Generates parametric maps for midband fit, spectral slope, and spectral intercept.
        """
        if not len(self.utcAnalysis.roiWindows):
            print("No analyzed windows to color")
            return
        if not len(self.utcAnalysis.ultrasoundImage.bmode.shape) == 3:
            self.convertImagesToRGB()
        if hasattr(self, 'paramaps'):
            del self.paramaps
        if hasattr(self, 'scParamaps'):
            del self.scParamaps
        
        bmodeShape = self.utcAnalysis.ultrasoundImage.bmode.shape
        self.windowIdxMap = np.zeros((bmodeShape[0], bmodeShape[1]), dtype=int)
        params = self.utcAnalysis.roiWindows[0].results.__dict__.keys()
        self.minParamVals = []; self.maxParamVals = []; self.paramaps = []
        self.params = []
        for param in params:
            if isinstance(getattr(self.utcAnalysis.roiWindows[0].results, param), (str, list, np.ndarray)):
                continue
            paramArr = [getattr(window.results, param) for window in self.utcAnalysis.roiWindows]
            self.minParamVals.append(min(paramArr))
            self.maxParamVals.append(max(paramArr))
            self.paramaps.append(self.utcAnalysis.ultrasoundImage.bmode.copy())
            self.params.append(param)
            
            for i, window in enumerate(self.utcAnalysis.roiWindows):
                paramColorIdx = int((255 / (self.maxParamVals[-1]-self.minParamVals[-1])
                                     )*(getattr(window.results, param)-self.minParamVals[-1])
                                    ) if self.minParamVals[-1] != self.maxParamVals[-1] else 125
        
                self.paramaps[-1][window.top: window.bottom+1, window.left: window.right+1] = self.cmaps[len(self.params)-1 % len(self.cmaps)][paramColorIdx]*255
                self.windowIdxMap[window.top: window.bottom+1, window.left: window.right+1] = i+1
                
        if hasattr(self.utcAnalysis.ultrasoundImage, 'scConfig'):
            self.scanConvertParamaps()

    def scanConvertRGB(self, image: np.ndarray) -> np.ndarray:
        """Converts a scan-converted grayscale image to RGB.

        Args:
            image (np.ndarray): Grayscale image to convert

        Returns:
            np.ndarray: RGB image
        """
        condensedIm = condenseArr(image)

        scConfig = self.utcAnalysis.ultrasoundImage.scConfig
        scBmode = self.utcAnalysis.ultrasoundImage.scBmode
        scStruct, _, _ = scanConvert(condensedIm, scConfig.width, scConfig.tilt,
                                        scConfig.startDepth, scConfig.endDepth, desiredHeight=scBmode.shape[0])

        return expandArr(scStruct.scArr)
    
    def scanConvertParamaps(self):
        """Scan converts the parametric maps to match the B-mode image.
        """
        if not len(self.paramaps):
            print("Generate cmaps first")
            return
        if hasattr(self, 'scParamaps'):
            del self.scParamaps
            
        self.scParamaps = [self.scanConvertRGB(paramap) for paramap in self.paramaps]

        scConfig = self.utcAnalysis.ultrasoundImage.scConfig
        scBmode = self.utcAnalysis.ultrasoundImage.scBmode
        scStruct, _, _ = scanConvert(self.windowIdxMap, scConfig.width, scConfig.tilt,
                                        scConfig.startDepth, scConfig.endDepth, desiredHeight=scBmode.shape[0])
        self.scWindowIdxMap = scStruct.scArr
        
    def formatParamaps(self):
        """Adds colorbars to the parametric maps."""
        paramaps = self.scParamaps if hasattr(self.utcAnalysis.ultrasoundImage, 'scConfig') else self.paramaps
        
        if hasattr(self, 'legendParamaps'):
            del self.legendParamaps
        self.legendParamaps = []
        
        for paramIx, paramap in enumerate(paramaps):
            fig = plt.figure()
            gs = GridSpec(1, 2, width_ratios=[20, 1])  # Adjust width ratios as needed

            # Main image subplot
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(paramap, aspect="auto") # TODO: fix
            ax.axis('off')

            # Create a separate mappable for the colorbar
            norm = mpl.colors.Normalize(vmin=0, vmax=255)
            cmapMappable = mpl.cm.ScalarMappable(norm=norm, cmap=self.cmapNames[paramIx % len(self.cmapNames)])

            # Colorbar subplot with exact same height
            cax = fig.add_subplot(gs[0, 1])
            cbar = plt.colorbar(cmapMappable, cax=cax, orientation='vertical')
            customTickLocations = [0, 255/5, 2*255/5, 3*255/5, 4*255/5, 255]  # Example: min, middle, max
            minVal = self.minParamVals[paramIx]; maxVal = self.maxParamVals[paramIx]
            valRange = maxVal - minVal
            customTickLabels = [np.round(minVal, decimals=2), 
                                np.round(minVal + valRange/5, decimals=2), 
                                np.round(minVal + 2*valRange/5, decimals=2),
                                np.round(minVal + 3*valRange/5, decimals=2), 
                                np.round(minVal + 4*valRange/5, decimals=2), 
                                np.round(maxVal, decimals=2)]       # Example: custom labels

            cbar.set_ticks(customTickLocations)
            cbar.set_ticklabels(customTickLabels)
            cbar.set_label(self.params[paramIx], fontweight='bold', fontsize=14)
            fig.tight_layout()
            self.legendParamaps.append(fig)


    def plotPsData(self) -> plt.Figure:
        """Plots the power spectrum data for each window in the ROI.
        
        The power spectrum data is plotted along with the average power spectrum and a line of best fit
        used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
        frequency band used for analysis.
        """
        ssArr = [window.results.ss for window in self.utcAnalysis.roiWindows]
        siArr = [window.results.si for window in self.utcAnalysis.roiWindows]
        npsArr = [window.results.nps for window in self.utcAnalysis.roiWindows]
        
        fig, ax = plt.subplots()

        ssMean = np.mean(np.array(ssArr)/1e6)
        siMean = np.mean(siArr)
        npsArr = [window.results.nps for window in self.utcAnalysis.roiWindows]
        avNps = np.mean(npsArr, axis=0)
        f = self.utcAnalysis.roiWindows[0].results.f
        x = np.linspace(min(f), max(f), 100)
        y = ssMean*x + siMean

        for nps in npsArr[:-1]:
            ax.plot(f/1e6, nps, c="b", alpha=0.2)
        ax.plot(f/1e6, npsArr[-1], c="b", alpha=0.2, label="Window NPS")
        ax.plot(f/1e6, avNps, color="r", label="Av NPS")
        ax.plot(x/1e6, y, c="orange", label="Av LOBF")
        ax.plot(2*[self.utcAnalysis.config.analysisFreqBand[0]/1e6], [np.amin(npsArr), np.amax(npsArr)], c="purple")
        ax.plot(2*[self.utcAnalysis.config.analysisFreqBand[1]/1e6], [np.amin(npsArr), np.amax(npsArr)], c="purple", label="Analysis Band")
        ax.set_title("Normalized Power Spectra")
        ax.legend()
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Power (dB)")
        ax.set_ylim([np.amin(npsArr), np.amax(npsArr)])
        ax.set_xlim([min(f)/1e6, max(f)/1e6])
        return fig
