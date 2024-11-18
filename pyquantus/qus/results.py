from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from pyquantus.qus.analysis import SpectralAnalysis
from pyquantus.qus.transforms import condenseArr, expandArr
from pyquantus.parse.objects import ScConfig
from pyquantus.parse.transforms import scanConvert

class SpectralData:
    """Class to store spectral data and images after QUS analysis.
    
    This class supports both scan converted and non-scan converted images. 
    Once analysis is completed in the SpectralAnalysis class, the results are 
    stored in this class for further evaluation and visualization.
    
    Attributes:
        spectralAnalysis (SpectralAnalysis): Spectral analysis object
        depth (float): Depth of the image in mm
        width (float): Width of the image in mm
        roiWidthScale (int): Width of the ROI window in pixels
        roiDepthScale (int): Depth of the ROI window in pixels
        rectCoords (List[int]): Coordinates of each window in the ROI
        mbfIm (np.ndarray): Midband fit parametric map
        ssIm (np.ndarray): Spectral slope parametric map
        siIm (np.ndarray): Spectral intercept parametric map
        scMbfIm (np.ndarray): Scan converted midband fit parametric map
        scSsIm (np.ndarray): Scan converted spectral slope parametric map
        scSiIm (np.ndarray): Scan converted spectral intercept parametric map
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
    """
    def __init__(self):
        self.spectralAnalysis: SpectralAnalysis
        self.depth: float # mm
        self.width: float # mm
        self.roiWidthScale: int
        self.roiDepthScale: int
        self.rectCoords: List[int]
        
        self.mbfIm: np.ndarray
        self.ssIm: np.ndarray
        self.siIm: np.ndarray
        self.scMbfIm: np.ndarray
        self.scSsIm: np.ndarray
        self.scSiIm: np.ndarray

        self.minMbf: float; self.maxMbf: float; self.mbfArr: List[float]
        self.minSs: float; self.maxSs: float; self.ssArr: List[float]
        self.minSi: float; self.maxSi: float; self.siArr: List[float]

        self.scConfig: ScConfig
        self.mbfCmap: list = plt.get_cmap("viridis").colors #type: ignore
        self.ssCmap: list = plt.get_cmap("magma").colors #type: ignore
        self.siCmap: list = plt.get_cmap("plasma").colors #type: ignore

    def convertImagesToRGB(self):
        """Converts grayscale images to RGB for colormap application.
        """
        self.spectralAnalysis.ultrasoundImage.bmode = cv2.cvtColor(
            np.array(self.spectralAnalysis.ultrasoundImage.bmode).astype('uint8'),
            cv2.COLOR_GRAY2RGB
        )
        if hasattr(self.spectralAnalysis.ultrasoundImage, 'scBmode'):
            self.spectralAnalysis.ultrasoundImage.scBmode = cv2.cvtColor(
                np.array(self.spectralAnalysis.ultrasoundImage.scBmode).astype('uint8'),
                cv2.COLOR_GRAY2RGB
            )

    def drawCmaps(self):
        """Generates parametric maps for midband fit, spectral slope, and spectral intercept.
        """
        if not len(self.spectralAnalysis.roiWindows):
            print("No analyzed windows to color")
            return
        
        self.mbfArr = [window.results.mbf for window in self.spectralAnalysis.roiWindows]
        self.minMbf = min(self.mbfArr); self.maxMbf = max(self.mbfArr)
        self.ssArr = [window.results.ss for window in self.spectralAnalysis.roiWindows]
        self.minSs = min(self.ssArr); self.maxSs = max(self.ssArr)
        self.siArr = [window.results.si for window in self.spectralAnalysis.roiWindows]
        self.minSi = min(self.siArr); self.maxSi = max(self.siArr)

        if not len(self.spectralAnalysis.ultrasoundImage.bmode.shape) == 3:
            self.convertImagesToRGB()
        self.mbfIm = self.spectralAnalysis.ultrasoundImage.bmode.copy()
        self.ssIm = self.mbfIm.copy(); self.siIm = self.ssIm.copy()

        for window in self.spectralAnalysis.roiWindows:
            mbfColorIdx = int((255 / (self.maxMbf-self.minMbf))*(window.results.mbf-self.minMbf)) if self.minMbf != self.maxMbf else 125
            ssColorIdx = int((255 / (self.maxSs-self.minSs))*(window.results.ss-self.minSs)) if self.minSs != self.maxSs else 125
            siColorIdx = int((255 / (self.maxSi-self.minSi))*(window.results.si-self.minSi)) if self.minSi != self.maxSi else 125
            self.mbfIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.mbfCmap[mbfColorIdx])*255
            self.ssIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.ssCmap[ssColorIdx])*255
            self.siIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.siCmap[siColorIdx])*255

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

    def plotPsData(self):
        """Plots the power spectrum data for each window in the ROI.
        
        The power spectrum data is plotted along with the average power spectrum and a line of best fit
        used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
        frequency band used for analysis.
        """
        _, ax = plt.subplots()

        ssMean = np.mean(self.ssArr)
        siMean = np.mean(self.siArr)
        npsArr = [window.results.nps for window in self.spectralAnalysis.roiWindows]
        avNps = np.mean(npsArr, axis=0)
        f = self.spectralAnalysis.roiWindows[0].results.f
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
        assert len(self.spectralAnalysis.ultrasoundImage.bmode.shape) == 3
        return self.spectralAnalysis.ultrasoundImage.bmode
    
    @bmode.setter
    def bmode(self, value: np.ndarray):
        """Setter for RGB B-mode image (no scan conversion).
        """
        self.spectralAnalysis.ultrasoundImage.bmode = value
    
    @property
    def scBmode(self):
        """Getter for scan converted RGB B-mode image.
        """
        assert len(self.spectralAnalysis.ultrasoundImage.scBmode.shape) == 3
        return self.spectralAnalysis.ultrasoundImage.scBmode
    
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
            self.spectralAnalysis.ultrasoundImage.scBmode = value
        else:
            self.spectralAnalysis.ultrasoundImage.bmode = value
    
    @property
    def splineX(self):
        """Getter for spline X coordinates regardless of scan conversion.
        """
        if hasattr(self, "scConfig"):
            return self.spectralAnalysis.scSplineX
        return self.spectralAnalysis.splineX
    
    @splineX.setter
    def splineX(self, value: np.ndarray):
        """Setter for spline X coordinates regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            self.spectralAnalysis.scSplineX = value
        else:
            self.spectralAnalysis.splineX = value

    @property
    def splineY(self):
        """Getter for spline Y coordinates regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.spectralAnalysis.scSplineY
        return self.spectralAnalysis.splineY
    
    @splineY.setter
    def splineY(self, value: np.ndarray):
        """Setter for spline Y coordinates regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            self.spectralAnalysis.scSplineY = value
        else:
            self.spectralAnalysis.splineY = value
    
    @property
    def waveLength(self):
        """Getter for wavelength of the ultrasound signal stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.waveLength
    
    @property
    def axWinSize(self):
        """Getter for axial window size stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.axWinSize
    
    @axWinSize.setter
    def axWinSize(self, value: float):
        """Setter for axial window size stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.axWinSize = value

    @property
    def latWinSize(self):
        """Getter for lateral window size stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.latWinSize
    
    @latWinSize.setter
    def latWinSize(self, value: float):
        """Setter for lateral window size stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.latWinSize = value

    @property
    def axOverlap(self):
        """Getter for axial overlap stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.axialOverlap
    
    @axOverlap.setter
    def axOverlap(self, value: float):
        """Setter for axial overlap stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.axialOverlap = value
    
    @property
    def latOverlap(self):
        """Getter for lateral overlap stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.lateralOverlap
    
    @latOverlap.setter
    def latOverlap(self, value: float):
        """Setter for lateral overlap stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.lateralOverlap = value
    
    @property
    def roiWindowThreshold(self):
        """Getter for ROI window threshold stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.windowThresh
    
    @roiWindowThreshold.setter
    def roiWindowThreshold(self, value: float):
        """Setter for ROI window threshold stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.windowThresh = value
    
    @property
    def analysisFreqBand(self):
        """Getter for analysis frequency band stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.analysisFreqBand
    
    @analysisFreqBand.setter
    def analysisFreqBand(self, value: List[int]):
        """Setter for analysis frequency band stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.analysisFreqBand = value

    @property
    def transducerFreqBand(self):
        """Getter for transducer frequency band stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.transducerFreqBand
    
    @transducerFreqBand.setter
    def transducerFreqBand(self, value: List[int]):
        """Setter for transducer frequency band stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.transducerFreqBand = value
    
    @property
    def samplingFrequency(self):
        """Getter for sampling frequency stored in the SpectralAnalysis class."""
        return self.spectralAnalysis.config.samplingFrequency
    
    @samplingFrequency.setter
    def samplingFrequency(self, value: int):
        """Setter for sampling frequency stored in the SpectralAnalysis class."""
        self.spectralAnalysis.config.samplingFrequency = value
    
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