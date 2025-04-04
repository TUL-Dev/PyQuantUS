from typing import List

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyquantus.utc.analysis import UtcAnalysis
from pyquantus.utc.transforms import condenseArr, expandArr
from pyquantus.parse.objects import ScConfig
from pyquantus.parse.transforms import scanConvert

class UtcData:
    """Class to store UTC data and images after analysis.
    
    This class supports both scan converted and non-scan converted images. 
    Once analysis is completed in the UtcAnalysis class, the results are 
    stored in this class for further evaluation and visualization.
    
    Attributes:
        utcAnalysis (UtcAnalysis): UTC analysis object
        depth (float): Depth of the image in mm
        width (float): Width of the image in mm
        roiWidthScale (int): Width of the ROI window in pixels
        roiDepthScale (int): Depth of the ROI window in pixels
        rectCoords (List[int]): Coordinates of each window in the ROI
        mbfIm (np.ndarray): Midband fit parametric map
        ssIm (np.ndarray): Spectral slope parametric map
        siIm (np.ndarray): Spectral intercept parametric map
        attCoef (np.ndarray): Attenuation coefficient parametric map
        bscIm (np.ndarray): Backscatter coefficient parametric map
        uNakagamiIm (np.ndarray): Nakagami shape parameter parametric map
        scMbfIm (np.ndarray): Scan converted midband fit parametric map
        scSsIm (np.ndarray): Scan converted spectral slope parametric map
        scSiIm (np.ndarray): Scan converted spectral intercept parametric map
        scAttCoefIm (np.ndarray): Scan converted attenuation coefficient parametric map
        scBscIm (np.ndarray): Scan converted backscatter coefficient parametric map
        scUNakagamiIm (np.ndarray): Scan converted Nakagami shape parameter parametric map
        minMbf (float): Minimum midband fit value in ROI
        maxMbf (float): Maximum midband fit value in ROI
        mbfArr (List[float]): Midband fit values for each window in ROI
        minSs (float): Minimum spectral slope value in ROI
        maxSs (float): Maximum spectral slope value in ROI
        ssArr (List[float]): Spectral slope values for each window in ROI
        minSi (float): Minimum spectral intercept value in ROI
        maxSi (float): Maximum spectral intercept value in ROI
        siArr (List[float]): Spectral intercept values for each window in ROI
        scConfig (ScConfig): Scan conversion configuration
        mbfCmap (list): Midband fit colormap used for parametric maps
        ssCmap (list): Spectral slope colormap used for parametric maps
        siCmap (list): Spectral intercept colormap used for parametric maps        
        attCoefCmap (list): Attenuation coefficient colormap used for parametric maps
        bscCmap (list): Backscatter coefficient colormap used for parametric maps
        uNakagamiCmap (list): Nakagami shape parameter colormap used for parametric maps
    """
    def __init__(self):
        self.utcAnalysis: UtcAnalysis
        self.depth: float # mm
        self.width: float # mm
        self.roiWidthScale: int
        self.roiDepthScale: int
        self.rectCoords: List[int]
        
        self.mbfIm: np.ndarray
        self.ssIm: np.ndarray
        self.siIm: np.ndarray
        self.attCoefIm: np.ndarray
        self.bscIm: np.ndarray
        self.uNakagamiIm: np.ndarray
        self.windowIdxMap: np.ndarray
        self.scMbfIm: np.ndarray
        self.scSsIm: np.ndarray
        self.scSiIm: np.ndarray
        self.scAttCoefIm: np.ndarray
        self.scBscIm: np.ndarray
        self.scUNakagamiIm: np.ndarray
        self.scWindowIdxMap: np.ndarray
        self.cbarParamaps: List[plt.Figure] = []

        self.minMbf: float; self.maxMbf: float; self.mbfArr: List[float]
        self.minSs: float; self.maxSs: float; self.ssArr: List[float]
        self.minSi: float; self.maxSi: float; self.siArr: List[float]

        self.scConfig: ScConfig
        self.mbfCmap: list = plt.get_cmap("viridis").colors #type: ignore
        self.ssCmap: list = plt.get_cmap("magma").colors #type: ignore
        self.siCmap: list = plt.get_cmap("plasma").colors #type: ignore
        self.attCoefCmap: list = plt.get_cmap("inferno").colors #type: ignore
        self.bscCmap: list = plt.get_cmap("cividis").colors #type: ignore
        
        summerCmap = plt.get_cmap("summer")
        self.uNakagamiCmap: list = [summerCmap(i)[:3] for i in range(summerCmap.N)]

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

    def drawCmaps(self):
        """Generates parametric maps for midband fit, spectral slope, and spectral intercept.
        """
        if not len(self.utcAnalysis.roiWindows):
            print("No analyzed windows to color")
            return
        
        self.mbfArr = [window.results.mbf for window in self.utcAnalysis.roiWindows]
        self.minMbf = min(self.mbfArr); self.maxMbf = max(self.mbfArr)
        self.ssArr = [window.results.ss for window in self.utcAnalysis.roiWindows]
        self.minSs = min(self.ssArr); self.maxSs = max(self.ssArr)
        self.siArr = [window.results.si for window in self.utcAnalysis.roiWindows]
        self.minSi = min(self.siArr); self.maxSi = max(self.siArr)
        # self.attCoefArr = [window.results.attCoef for window in self.utcAnalysis.roiWindows]
        # self.minAttCoef = min(self.attCoefArr); self.maxAttCoef = max(self.attCoefArr)
        # self.bscArr = [window.results.bsc for window in self.utcAnalysis.roiWindows]
        # self.minBsc = min(self.bscArr); self.maxBsc = max(self.bscArr)
        # self.uNakagamiArr = [window.results.uNakagami for window in self.utcAnalysis.roiWindows]
        # self.minUNakagami = min(self.uNakagamiArr); self.maxUNakagami = max(self.uNakagamiArr)

        if not len(self.utcAnalysis.ultrasoundImage.bmode.shape) == 3:
            self.convertImagesToRGB()
        self.mbfIm = self.utcAnalysis.ultrasoundImage.bmode.copy()
        self.ssIm = self.mbfIm.copy(); self.siIm = self.ssIm.copy()
        # self.attCoefIm = self.ssIm.copy(); self.bscIm = self.ssIm.copy(); self.uNakagamiIm = self.ssIm.copy()
        self.windowIdxMap = np.zeros((self.mbfIm.shape[0], self.mbfIm.shape[1])).astype(int)

        for i, window in enumerate(self.utcAnalysis.roiWindows):
            mbfColorIdx = int((255 / (self.maxMbf-self.minMbf))*(window.results.mbf-self.minMbf)) if self.minMbf != self.maxMbf else 125
            ssColorIdx = int((255 / (self.maxSs-self.minSs))*(window.results.ss-self.minSs)) if self.minSs != self.maxSs else 125
            siColorIdx = int((255 / (self.maxSi-self.minSi))*(window.results.si-self.minSi)) if self.minSi != self.maxSi else 125
            self.mbfIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.mbfCmap[mbfColorIdx])*255
            self.ssIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.ssCmap[ssColorIdx])*255
            self.siIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.siCmap[siColorIdx])*255
            # self.attCoefIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.attCoefCmap[mbfColorIdx])*255
            # self.bscIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.bscCmap[mbfColorIdx])*255
            # self.uNakagamiIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.uNakagamiCmap[mbfColorIdx])*255
            # self.windowIdxMap[window.top: window.bottom+1, window.left: window.right+1] = i+1

    def scanConvertRGB(self, image: np.ndarray) -> np.ndarray:
        """Converts a scan-converted grayscale image to RGB.

        Args:
            image (np.ndarray): Grayscale image to convert

        Returns:
            np.ndarray: RGB image
        """
        condensedIm = condenseArr(image)

        scStruct, _, _ = scanConvert(condensedIm, self.scConfig.width, self.scConfig.tilt,
                                        self.scConfig.startDepth, self.scConfig.endDepth, desiredHeight=self.scBmode.shape[0])

        return expandArr(scStruct.scArr)
    
    def scanConvertCmaps(self):
        """Scan converts the parametric maps to match the B-mode image.
        """
        if self.mbfIm is None:
            print("Generate cmaps first")
            return
        
        self.scMbfIm = self.scanConvertRGB(self.mbfIm)
        self.scSsIm = self.scanConvertRGB(self.ssIm)
        self.scSiIm = self.scanConvertRGB(self.siIm)
        # self.scAttCoefIm = self.scanConvertRGB(self.attCoefIm)
        # self.scBscIm = self.scanConvertRGB(self.bscIm)
        # self.scUNakagamiIm = self.scanConvertRGB(self.uNakagamiIm)

        scStruct, _, _ = scanConvert(self.windowIdxMap, self.scConfig.width, self.scConfig.tilt,
                                        self.scConfig.startDepth, self.scConfig.endDepth, desiredHeight=self.scBmode.shape[0])
        self.scWindowIdxMap = scStruct.scArr

    def formatParamaps(self):
        """Adds colorbars to the parametric maps."""
        if hasattr(self, "scConfig"):
            paramaps = [self.scMbfIm, self.scSsIm, self.scSiIm, self.scAttCoefIm, self.scBscIm, self.scUNakagamiIm]
        else:
            paramaps = [self.mbfIm, self.ssIm, self.siIm, self.attCoefIm, self.bscIm, self.uNakagamiIm]
        minParamVals = [self.minMbf, self.minSs, self.minSi, self.minAttCoef, self.minBsc, self.minUNakagami]
        maxParamVals = [self.maxMbf, self.maxSs, self.maxSi, self.maxAttCoef, self.maxBsc, self.maxUNakagami]
        paramNames = ["mbf (dB)", "ss (dB/MHz)", "si (dB)", "attCoef (dB/cm/MHz)", "bsc (1/cm-sr)", "uNakagami (shape)"]
        cmaps = [self.mbfCmap, self.ssCmap, self.siCmap, self.attCoefCmap, self.bscCmap, self.uNakagamiCmap]
        
        for paramIx, paramap in enumerate(paramaps):
            fig = plt.figure()
            gs = GridSpec(1, 2, width_ratios=[20, 1])  # Adjust width ratios as needed

            # Main image subplot
            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(paramap, aspect="auto") # TODO: fix
            ax.axis('off')

            # Create a separate mappable for the colorbar
            norm = mpl.colors.Normalize(vmin=0, vmax=255)
            cmapMappable = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colors.ListedColormap(cmaps[paramIx]))

            # Colorbar subplot with exact same height
            cax = fig.add_subplot(gs[0, 1])
            cbar = plt.colorbar(cmapMappable, cax=cax, orientation='vertical')
            customTickLocations = [0, 255/5, 2*255/5, 3*255/5, 4*255/5, 255]  # Example: min, middle, max
            minVal = minParamVals[paramIx]; maxVal = maxParamVals[paramIx]
            valRange = maxVal - minVal
            customTickLabels = [np.round(minVal, decimals=2), 
                                np.round(minVal + valRange/5, decimals=2), 
                                np.round(minVal + 2*valRange/5, decimals=2),
                                np.round(minVal + 3*valRange/5, decimals=2), 
                                np.round(minVal + 4*valRange/5, decimals=2), 
                                np.round(maxVal, decimals=2)]       # Example: custom labels

            cbar.set_ticks(customTickLocations)
            cbar.set_ticklabels(customTickLabels)
            cbar.set_label(paramNames[paramIx], fontweight='bold', fontsize=14)
            fig.tight_layout()
            self.cbarParamaps.append(fig)

    def plotPsData(self):
        """Plots the power spectrum data for each window in the ROI.
        
        The power spectrum data is plotted along with the average power spectrum and a line of best fit
        used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
        frequency band used for analysis.
        """
        _, ax = plt.subplots()

        ssMean = np.mean(np.array(self.ssArr)/1e6)
        siMean = np.mean(self.siArr)
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
        ax.plot(2*[self.analysisFreqBand[0]/1e6], [np.amin(npsArr), np.amax(npsArr)], c="purple")
        ax.plot(2*[self.analysisFreqBand[1]/1e6], [np.amin(npsArr), np.amax(npsArr)], c="purple", label="Analysis Band")
        plt.legend()
        plt.show()

    @property
    def bmode(self) -> np.ndarray:
        """Getter for RGB B-mode image (no scan conversion).
        """
        assert len(self.utcAnalysis.ultrasoundImage.bmode.shape) == 3
        return self.utcAnalysis.ultrasoundImage.bmode
    
    @bmode.setter
    def bmode(self, value: np.ndarray):
        """Setter for RGB B-mode image (no scan conversion).
        """
        self.utcAnalysis.ultrasoundImage.bmode = value
    
    @property
    def scBmode(self):
        """Getter for scan converted RGB B-mode image.
        """
        assert len(self.utcAnalysis.ultrasoundImage.scBmode.shape) == 3
        return self.utcAnalysis.ultrasoundImage.scBmode
    
    @property
    def finalBmode(self) -> np.ndarray:
        """Getter for RGB B-mode image regardless of scan conversion.
        """
        if hasattr(self, "scConfig"):
            return self.scBmode
        return self.bmode
    
    @finalBmode.setter
    def finalBmode(self, value: np.ndarray):
        """Setter for RGB B-mode image regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            self.utcAnalysis.ultrasoundImage.scBmode = value
        else:
            self.utcAnalysis.ultrasoundImage.bmode = value
    
    @property
    def splineX(self):
        """Getter for spline X coordinates regardless of scan conversion.
        """
        if hasattr(self, "scConfig"):
            return self.utcAnalysis.scSplineX
        return self.utcAnalysis.splineX
    
    @splineX.setter
    def splineX(self, value: np.ndarray):
        """Setter for spline X coordinates regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            self.utcAnalysis.scSplineX = value
        else:
            self.utcAnalysis.splineX = value

    @property
    def splineY(self):
        """Getter for spline Y coordinates regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.utcAnalysis.scSplineY
        return self.utcAnalysis.splineY
    
    @splineY.setter
    def splineY(self, value: np.ndarray):
        """Setter for spline Y coordinates regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            self.utcAnalysis.scSplineY = value
        else:
            self.utcAnalysis.splineY = value
    
    @property
    def waveLength(self):
        """Getter for wavelength of the ultrasound signal stored in the UtcAnalysis class."""
        return self.utcAnalysis.waveLength
    
    @property
    def axWinSize(self):
        """Getter for axial window size stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.axWinSize
    
    @axWinSize.setter
    def axWinSize(self, value: float):
        """Setter for axial window size stored in the UtcAnalysis class."""
        self.utcAnalysis.config.axWinSize = value

    @property
    def latWinSize(self):
        """Getter for lateral window size stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.latWinSize
    
    @latWinSize.setter
    def latWinSize(self, value: float):
        """Setter for lateral window size stored in the UtcAnalysis class."""
        self.utcAnalysis.config.latWinSize = value

    @property
    def axOverlap(self):
        """Getter for axial overlap stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.axialOverlap
    
    @axOverlap.setter
    def axOverlap(self, value: float):
        """Setter for axial overlap stored in the UtcAnalysis class."""
        self.utcAnalysis.config.axialOverlap = value
    
    @property
    def latOverlap(self):
        """Getter for lateral overlap stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.lateralOverlap
    
    @latOverlap.setter
    def latOverlap(self, value: float):
        """Setter for lateral overlap stored in the UtcAnalysis class."""
        self.utcAnalysis.config.lateralOverlap = value
    
    @property
    def roiWindowThreshold(self):
        """Getter for ROI window threshold stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.windowThresh
    
    @roiWindowThreshold.setter
    def roiWindowThreshold(self, value: float):
        """Setter for ROI window threshold stored in the UtcAnalysis class."""
        self.utcAnalysis.config.windowThresh = value
    
    @property
    def analysisFreqBand(self):
        """Getter for analysis frequency band stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.analysisFreqBand
    
    @analysisFreqBand.setter
    def analysisFreqBand(self, value: List[int]):
        """Setter for analysis frequency band stored in the UtcAnalysis class."""
        self.utcAnalysis.config.analysisFreqBand = value

    @property
    def transducerFreqBand(self):
        """Getter for transducer frequency band stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.transducerFreqBand
    
    @transducerFreqBand.setter
    def transducerFreqBand(self, value: List[int]):
        """Setter for transducer frequency band stored in the UtcAnalysis class."""
        self.utcAnalysis.config.transducerFreqBand = value
    
    @property
    def samplingFrequency(self):
        """Getter for sampling frequency stored in the UtcAnalysis class."""
        return self.utcAnalysis.config.samplingFrequency
    
    @samplingFrequency.setter
    def samplingFrequency(self, value: int):
        """Setter for sampling frequency stored in the UtcAnalysis class."""
        self.utcAnalysis.config.samplingFrequency = value
    
    @property
    def pixWidth(self):
        """Getter for pixel width of the B-mode image regardless of scan conversion."""
        return self.finalBmode.shape[1]
    
    @property
    def pixDepth(self):
        """Getter for pixel depth of the B-mode image regardless of scan conversion."""
        return self.finalBmode.shape[0]

    @property
    def numSamplesDrOut(self):
        """Getter for number of samples in the scan converted image (used for Canon 
        project-specific applications)."""
        return self.scConfig.numSamplesDrOut
    
    @property
    def lateralRes(self):
        """Getter for lateral resolution of the image regardless of scan conversion."""
        return self.width / self.finalBmode.shape[1]
    
    @property
    def axialRes(self):
        """Getter for axial resolution of the final image regardless of scan conversion."""
        return self.depth / self.finalBmode.shape[0]
    
    @property
    def finalMbfIm(self):
        """Getter for final midband fit parametric map regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scMbfIm
        return self.mbfIm

    @property
    def finalSsIm(self):
        """Getter for final spectral slope parametric map regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scSsIm
        return self.ssIm
    
    @property
    def finalSiIm(self):
        """Getter for final spectral intercept parametric map regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scSiIm
        return self.siIm
    
    @property
    def finalAttCoefIm(self):
        """Getter for final attenuation coefficient parametric map regagrdless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scAttCoefIm
        return self.attCoefIm
    
    @property
    def finalBscIm(self):
        """Getter for final backscatter coefficient parametric map regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scBscIm
        return self.bscIm
    
    @property
    def finalUNakagamiIm(self):
        """Getter for final Nakagami shape parameter parametric map regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scUNakagamiIm
        return self.uNakagamiIm
    
    @property
    def finalWindowIdxMap(self):
        """Getter for final window index map regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scWindowIdxMap
        return self.windowIdxMap