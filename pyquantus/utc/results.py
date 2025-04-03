from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pyquantus.utc.analysis3d import UtcAnalysis3d
from pyquantus.parse.objects import ScParams
from pyquantus.parse.transforms import scanConvert3dVolumeSeries

class UtcData3D:
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
        scParams (ScParams): 3D scan conversion parameters
        mbfCmap (list): Midband fit colormap used for parametric maps
        ssCmap (list): Spectral slope colormap used for parametric maps
        siCmap (list): Spectral intercept colormap used for parametric maps        
    """
    def __init__(self):
        self.utcAnalysis: UtcAnalysis3d
        self.depth: float # mm
        self.width: float # mm
        self.roiWidthScale: int
        self.roiDepthScale: int
        self.rectCoords: List[int]
        
        self.mbfIm: np.ndarray
        self.ssIm: np.ndarray
        self.siIm: np.ndarray
        self.windowIdxMap: np.ndarray
        self.scMbfIm: np.ndarray
        self.scSsIm: np.ndarray
        self.scSiIm: np.ndarray
        self.scWindowIdxMap: np.ndarray

        self.minMbf: float; self.maxMbf: float; self.mbfArr: List[float]
        self.minSs: float; self.maxSs: float; self.ssArr: List[float]
        self.minSi: float; self.maxSi: float; self.siArr: List[float]

        self.scParams: ScParams
        self.mbfCmap: list = plt.get_cmap("viridis").colors #type: ignore
        self.ssCmap: list = plt.get_cmap("magma").colors #type: ignore
        self.siCmap: list = plt.get_cmap("plasma").colors #type: ignore

    def convertImagesToRGB(self):
        """Converts grayscale images to RGB for colormap application.
        """
        self.utcAnalysis.ultrasoundImage.bmode -= np.amin(self.utcAnalysis.ultrasoundImage.bmode)
        self.utcAnalysis.ultrasoundImage.bmode *= 255/np.amax(self.utcAnalysis.ultrasoundImage.bmode)
        self.utcAnalysis.ultrasoundImage.bmode = np.array([cv2.cvtColor(
            np.array(self.utcAnalysis.ultrasoundImage.bmode[i]).astype('uint8'),
            cv2.COLOR_GRAY2RGB) for i in range(self.utcAnalysis.ultrasoundImage.bmode.shape[0])
        ])
        if hasattr(self.utcAnalysis.ultrasoundImage, 'scBmode'):
            self.utcAnalysis.ultrasoundImage.scBmode -= np.amin(self.utcAnalysis.ultrasoundImage.scBmode)
            self.utcAnalysis.ultrasoundImage.scBmode *= 255/np.amax(self.utcAnalysis.ultrasoundImage.scBmode)
            self.utcAnalysis.ultrasoundImage.scBmode = np.array([cv2.cvtColor(
                np.array(self.utcAnalysis.ultrasoundImage.scBmode[i]).astype('uint8'),
                cv2.COLOR_GRAY2RGB) for i in range(self.utcAnalysis.ultrasoundImage.scBmode.shape[0])
])

    def drawCmaps(self):
        """Generates parametric maps for midband fit, spectral slope, and spectral intercept.
        """
        if not len(self.utcAnalysis.voiWindows):
            print("No analyzed windows to color")
            return
        
        self.mbfArr = [window.results.mbf for window in self.utcAnalysis.voiWindows]
        self.minMbf = min(self.mbfArr); self.maxMbf = max(self.mbfArr)
        self.ssArr = [window.results.ss for window in self.utcAnalysis.voiWindows]
        self.minSs = min(self.ssArr); self.maxSs = max(self.ssArr)
        self.siArr = [window.results.si for window in self.utcAnalysis.voiWindows]
        self.minSi = min(self.siArr); self.maxSi = max(self.siArr)

        if not len(self.utcAnalysis.ultrasoundImage.bmode.shape) == 4:
            self.convertImagesToRGB()
        self.mbfIm = self.utcAnalysis.ultrasoundImage.bmode.copy()
        self.ssIm = self.mbfIm.copy(); self.siIm = self.ssIm.copy()
        self.windowIdxMap = np.zeros((self.mbfIm.shape[0], self.mbfIm.shape[1], self.mbfIm.shape[2])).astype(int)

        for i, window in enumerate(self.utcAnalysis.voiWindows):
            mbfColorIdx = int((255 / (self.maxMbf-self.minMbf))*(window.results.mbf-self.minMbf)) if self.minMbf != self.maxMbf else 125
            ssColorIdx = int((255 / (self.maxSs-self.minSs))*(window.results.ss-self.minSs)) if self.minSs != self.maxSs else 125
            siColorIdx = int((255 / (self.maxSi-self.minSi))*(window.results.si-self.minSi)) if self.minSi != self.maxSi else 125
            self.mbfIm[window.corMin: window.corMax+1, window.latMin: window.latMax+1, window.axMin: window.axMax+1] = np.array(self.mbfCmap[mbfColorIdx])*255
            self.ssIm[window.corMin: window.corMax+1, window.latMin: window.latMax+1, window.axMin: window.axMax+1] = np.array(self.ssCmap[ssColorIdx])*255
            self.siIm[window.corMin: window.corMax+1, window.latMin: window.latMax+1, window.axMin: window.axMax+1] = np.array(self.siCmap[siColorIdx])*255
            self.windowIdxMap[window.corMin: window.corMax+1, window.latMin: window.latMax+1, window.axMin: window.axMax+1] = i+1
    
    def scanConvertCmaps(self):
        """Scan converts the parametric maps to match the B-mode image.
        """
        if self.mbfIm is None:
            print("Generate cmaps first")
            return
        self.scWindowIdxMap, _, _ = scanConvert3dVolumeSeries(self.windowIdxMap, self.scParams, scale=False, interp='nearest', normalize=False)
        self.scWindowIdxMap = self.scWindowIdxMap.astype(int)
        
        self.scMbfIm = self.utcAnalysis.ultrasoundImage.scBmode.copy()
        self.scSsIm = self.scMbfIm.copy(); self.scSiIm = self.scSsIm.copy()
        for i, window in tqdm(enumerate(self.utcAnalysis.voiWindows), total=len(self.utcAnalysis.voiWindows)):
            mbfColorIdx = int((255 / (self.maxMbf-self.minMbf))*(window.results.mbf-self.minMbf)) if self.minMbf != self.maxMbf else 125
            ssColorIdx = int((255 / (self.maxSs-self.minSs))*(window.results.ss-self.minSs)) if self.minSs != self.maxSs else 125
            siColorIdx = int((255 / (self.maxSi-self.minSi))*(window.results.si-self.minSi)) if self.minSi != self.maxSi else 125
            
            windowLoc = np.transpose(np.where(self.scWindowIdxMap == i+1))
            for loc in windowLoc:
                self.scMbfIm[loc[0], loc[1], loc[2]] = np.array(self.mbfCmap[mbfColorIdx])*255
                self.scSiIm[loc[0], loc[1], loc[2]] = np.array(self.siCmap[siColorIdx])*255
                self.scSsIm[loc[0], loc[1], loc[2]] = np.array(self.ssCmap[ssColorIdx])*255

    def plotPsData(self):
        """Plots the power spectrum data for each window in the ROI.
        
        The power spectrum data is plotted along with the average power spectrum and a line of best fit
        used for the midband fit, spectral slope, and spectral intercept calculations. Also plots the 
        frequency band used for analysis.
        """
        _, ax = plt.subplots()

        ssMean = np.mean(np.array(self.ssArr)/1e6)
        siMean = np.mean(self.siArr)
        npsArr = [window.results.nps for window in self.utcAnalysis.voiWindows]
        avNps = np.mean(npsArr, axis=0)
        f = self.utcAnalysis.voiWindows[0].results.f
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
        assert len(self.utcAnalysis.ultrasoundImage.bmode.shape) == 4
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
        assert len(self.utcAnalysis.ultrasoundImage.scBmode.shape) == 4
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
    def finalWindowIdxMap(self):
        """Getter for final window index map regardless of scan conversion."""
        if hasattr(self, "scConfig"):
            return self.scWindowIdxMap
        return self.windowIdxMap