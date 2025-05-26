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
from pyquantus.utc.analysis import Hscan

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        if self.utcAnalysis.computeExtraParamaps:
            self.attCoefArr = [window.results.attCoef for window in self.utcAnalysis.roiWindows]
            self.minAttCoef = min(self.attCoefArr); self.maxAttCoef = max(self.attCoefArr)
            self.bscArr = [window.results.bsc for window in self.utcAnalysis.roiWindows]
            self.minBsc = min(self.bscArr); self.maxBsc = max(self.bscArr)
            self.uNakagamiArr = [window.results.uNakagami for window in self.utcAnalysis.roiWindows]
            self.minUNakagami = min(self.uNakagamiArr); self.maxUNakagami = max(self.uNakagamiArr)

        if not len(self.utcAnalysis.ultrasoundImage.bmode.shape) == 3:
            self.convertImagesToRGB()
        self.mbfIm = self.utcAnalysis.ultrasoundImage.bmode.copy()
        self.ssIm = self.mbfIm.copy(); self.siIm = self.ssIm.copy()
        self.attCoefIm = self.ssIm.copy(); self.bscIm = self.ssIm.copy(); self.uNakagamiIm = self.ssIm.copy()
        self.windowIdxMap = np.zeros((self.mbfIm.shape[0], self.mbfIm.shape[1])).astype(int)

        for i, window in enumerate(self.utcAnalysis.roiWindows):
            mbfColorIdx = int((255 / (self.maxMbf-self.minMbf))*(window.results.mbf-self.minMbf)) if self.minMbf != self.maxMbf else 125
            ssColorIdx = int((255 / (self.maxSs-self.minSs))*(window.results.ss-self.minSs)) if self.minSs != self.maxSs else 125
            siColorIdx = int((255 / (self.maxSi-self.minSi))*(window.results.si-self.minSi)) if self.minSi != self.maxSi else 125
            self.mbfIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.mbfCmap[mbfColorIdx])*255
            self.ssIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.ssCmap[ssColorIdx])*255
            self.siIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.siCmap[siColorIdx])*255
            self.windowIdxMap[window.top: window.bottom+1, window.left: window.right+1] = i+1
            if self.utcAnalysis.computeExtraParamaps:
                attCoefColorIdx = int((255 / (self.maxAttCoef-self.minAttCoef))*(window.results.attCoef-self.minAttCoef)) if self.minAttCoef != self.maxAttCoef else 125
                bscColorIdx = int((255 / (self.maxBsc-self.minBsc))*(window.results.bsc-self.minBsc)) if self.minBsc != self.maxBsc else 125
                uNakagamiColorIdx = int((255 / (self.maxUNakagami-self.minUNakagami))*(window.results.uNakagami-self.minUNakagami)) if self.minUNakagami != self.maxUNakagami else 125
                self.attCoefIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.attCoefCmap[attCoefColorIdx])*255
                self.bscIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.bscCmap[bscColorIdx])*255
                self.uNakagamiIm[window.top: window.bottom+1, window.left: window.right+1] = np.array(self.uNakagamiCmap[uNakagamiColorIdx])*255

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
        self.scAttCoefIm = self.scanConvertRGB(self.attCoefIm)
        self.scBscIm = self.scanConvertRGB(self.bscIm)
        self.scUNakagamiIm = self.scanConvertRGB(self.uNakagamiIm)

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
    
    
    
###################################################################################
# H-scan post-processing
###################################################################################
class HscanPostProcessing:
    """Class for post-processing the H-scan data."""

    ###################################################################################
    # Constructor
    ###################################################################################
    def __init__(self,
                 hscan_obj: Hscan,
                 visualize: bool = False,
                 clip_fact: float = 1,
                 dyn_range: float = 100):
        """Constructor for the HscanPostProcessing class.
        
        Args:
            hscan_obj (Hscan): The Hscan object to be post-processed.
        """
        
        # input arguments
        self.hscan_obj = hscan_obj
        self.visualize = visualize
        self.clip_fact = clip_fact
        self.dyn_range = dyn_range
        
        # 1d
        self.signal_1d = None
        self.envelope_1d = None
        self.convolution_1d_1 = None
        self.convolution_envelope_1d_1 = None
        self.convolution_1d_2 = None
        self.convolution_envelope_1d_2 = None
        
        # 2d       
        self.signal_2d = None
        self.envelope_2d = None
        self.convolution_2d_1 = None
        self.convolution_envelope_2d_1 = None
        self.convolution_2d_2 = None
        self.convolution_envelope_2d_2 = None
        
        # run
        self.__run()
        
    ###################################################################################
    # Run
    ###################################################################################
    def __run(self):
        
        # set 1D signal and envelope
        self.set_1d_signal_and_envelope()
        
        # plot original signal 1D
        self.plot_original_signal_1d(self.signal_1d,
                                     self.envelope_1d,
                                     sampling_frequency = self.hscan_obj.wavelet_GHx_params_1['fs'])

        # plot GH1 wavelet
        self.hscan_obj.wavelet_GH1.visualize = True
        self.hscan_obj.wavelet_GH1.plot()
        
        # plot GH1 signal and envelope
        self.plot_hscan_signal_and_envelope(self.convolution_1d_1,
                                            self.convolution_envelope_1d_1,
                                            order = self.hscan_obj.wavelet_GHx_params_1['order'],
                                            sampling_frequency = self.hscan_obj.wavelet_GHx_params_1['fs'])   
        
        # plot GH2 wavelet
        self.hscan_obj.wavelet_GH2.visualize = True
        self.hscan_obj.wavelet_GH2.plot()

        # plot GH2 signal and envelope
        self.plot_hscan_signal_and_envelope(self.convolution_1d_2,
                                            self.convolution_envelope_1d_2,
                                            order = self.hscan_obj.wavelet_GHx_params_2['order'],
                                            sampling_frequency = self.hscan_obj.wavelet_GHx_params_2['fs'])  

        # set 2D signal and envelope
        self.set_2d_signal_and_envelope()

        # plot envelope
        self.image_envelope_2d(envelope_2d = self.envelope_2d,
                               title = "Original Envelope",
                               clip_fact = self.clip_fact,
                               dyn_range = self.dyn_range)
    
        # self.image_RB_difference_in_RGB_2d(R0B_final_2d = self.R0B_trimmed_3x2d, 
        #                                     additional_text="",
        #                                     log=False)
        
        # self.image_color_channel_2d(color_channel_2d = self.RGB_trimmed_3x2d, color="RGB", addtional_text="RGB_based on first frame", log=False)
        # self.image_color_channel_2d(color_channel_2d = self.G_channel_2d, color="green", addtional_text="single channel", log=False)
        # self.image_color_channel_2d(color_channel_2d = self.R_channel_2d, color="red",   addtional_text="single channel", log=False)
        # self.image_color_channel_2d(color_channel_2d = self.B_channel_2d, color="blue",  addtional_text="single channel", log=False)

        # # plot raw signal with wavelets
        # self.plot_1d_signal_envelope_and_fft(signal = self.df_trimmed_signal_1d,
        #                             envelope = self.df_trimmed_signal_envelope_1d)
        
        # self.plot_1d_signal_envelope_and_fft(signal = self.df_convolved_signal_with_ghx_1_1d,
        #                             envelope = self.df_convolved_signal_with_ghx_1_envelope_1d)
                
        # self.plot_1d_signal_envelope_and_fft(signal = self.df_convolved_signal_with_ghx_2_1d,
        #                             envelope = self.df_convolved_signal_with_ghx_2_envelope_1d)
        
        # self.plot_envelope(self.G_channel_2d[self.G_channel_2d.shape[0]//2, :], color = 'green')
        # self.plot_envelope(self.R_channel_2d[self.G_channel_2d.shape[0]//2, :], color = 'red')
        # self.plot_envelope(self.B_channel_2d[self.G_channel_2d.shape[0]//2, :], color = 'blue')

    

    ###################################################################################
    # Rotate and flip
    ###################################################################################
    @staticmethod
    def rotate_flip(
                    two_dimension_array: np.ndarray) -> np.ndarray:

        # Rotate the input array counterclockwise by 90 degrees
        rotated_array = np.rot90(two_dimension_array)
        
        # Flip the rotated array horizontally
        rotated_flipped_array = np.flipud(rotated_array)
        
        return rotated_flipped_array
    
    ###################################################################################
    # Set 1D signal and envelope
    ###################################################################################
    def set_1d_signal_and_envelope(self):
        """
        Extract a 1D signal and its envelope from the n-dimensional convolved H-scan data
        along the axis specified by self.hscan_obj.signal_axis, using index 0 for all other axes.
        """
        shape = self.hscan_obj.convolved_signal_with_ghx_1_nd.shape
        axis = self.hscan_obj.signal_axis

        # Build indices: 0 for all axes except the signal axis
        indices = []
        for i in range(len(shape)):
            if i == axis:
                indices.append(slice(None))
            else:
                indices.append(0)
        indices = tuple(indices)

        # Extract 1D signal and envelope
        self.convolution_1d_1 = self.hscan_obj.convolved_signal_with_ghx_1_nd[indices]
        self.convolution_envelope_1d_1 = self.hscan_obj.convolved_signal_with_ghx_1_envelope_nd[indices]

        self.convolution_1d_2 = self.hscan_obj.convolved_signal_with_ghx_2_nd[indices]
        self.convolution_envelope_1d_2 = self.hscan_obj.convolved_signal_with_ghx_2_envelope_nd[indices]
        
        self.signal_1d = self.hscan_obj.signal_nd[indices]
        self.envelope_1d = self.hscan_obj.envelope_nd[indices]

    ###################################################################################
    # Set 2D signal and envelope
    ###################################################################################
    def set_2d_signal_and_envelope(self):
        """
        Extract a 2D signal and its envelope from the n-dimensional convolved H-scan data
        along the axis specified by self.hscan_obj.signal_axis, self.hscan_obj.row_axis, using index 0 for all other axes.
        """
        logging.info("Starting extraction of 2D signal and envelope.")
        
        shape = self.hscan_obj.convolved_signal_with_ghx_1_nd.shape
        axis = self.hscan_obj.signal_axis
        row_axis = self.hscan_obj.row_axis
        
        logging.debug(f"Shape of convolved signal: {shape}")
        logging.debug(f"Signal axis: {axis}, Row axis: {row_axis}")
        
        # Build indices: 0 for all axes except the signal axis and row axis
        indices = []
        for i in range(len(shape)):
            if i == axis or i == row_axis:
                indices.append(slice(None))
            else:
                indices.append(0)
        indices = tuple(indices)
        
        logging.debug(f"Indices for extraction: {indices}")
        
        # Extract 2D signal and envelope
        self.convolution_2d_1 = self.hscan_obj.convolved_signal_with_ghx_1_nd[indices]
        self.convolution_envelope_2d_1 = self.hscan_obj.convolved_signal_with_ghx_1_envelope_nd[indices]
        
        self.convolution_2d_2 = self.hscan_obj.convolved_signal_with_ghx_2_nd[indices]
        self.convolution_envelope_2d_2 = self.hscan_obj.convolved_signal_with_ghx_2_envelope_nd[indices]
        
        self.signal_2d = self.hscan_obj.signal_nd[indices]
        self.envelope_2d = self.hscan_obj.envelope_nd[indices]
        
        # add logging of shapes
        logging.info(f"Shape of signal: {self.signal_2d.shape}")
        logging.info(f"Shape of envelope: {self.envelope_2d.shape}")
        logging.info(f"Shape of convolution_2d_1: {self.convolution_2d_1.shape}")
        logging.info(f"Shape of convolution_envelope_2d_1: {self.convolution_envelope_2d_1.shape}")
        logging.info(f"Shape of convolution_2d_2: {self.convolution_2d_2.shape}")
        logging.info(f"Shape of convolution_envelope_2d_2: {self.convolution_envelope_2d_2.shape}")
        
        logging.info("Completed extraction of 2D signal and envelope.")
    
    ###################################################################################
    # Plot H-scan signal and envelope
    ###################################################################################
    def plot_hscan_signal_and_envelope(self, signal_1d, envelope_1d, order: int, sampling_frequency: int):
        """Plot the H-scan signals and their envelopes.
        
        Args:
            signal_1d (np.ndarray): The 1D signal to plot
            envelope_1d (np.ndarray): The 1D envelope to plot
            order (int): The order of the Hermite polynomial used for the wavelet
            sampling_frequency (int): The sampling frequency of the signal

        This function creates a figure with subplots showing:
        1. GHx signal and its envelope
        2. FFT of GHx signal
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot GHx signal and envelope
        axes[0].plot(signal_1d, 'b-', label=f'GH{order} Signal', alpha=0.5)
        axes[0].plot(envelope_1d, 'r-', label=f'GH{order} Envelope')
        axes[0].set_title(f'GH{order} Signal and Envelope')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True)

        # Calculate and plot FFT of GHx signal
        fft_signal_1 = np.fft.fft(signal_1d)
        freq_signal_1 = np.fft.fftfreq(len(fft_signal_1), d=1/sampling_frequency)
        axes[1].plot(freq_signal_1[:len(freq_signal_1)//2] / 1e6, np.abs(fft_signal_1)[:len(fft_signal_1)//2], 'b-')
        axes[1].set_title(f'FFT of GH{order} Signal')
        axes[1].set_xlabel('Frequency [MHz]')
        axes[1].set_ylabel('Magnitude')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
        
    ###################################################################################
    # Plot original signal 1D
    ###################################################################################
    def plot_original_signal_1d(self, signal_1d, envelope_1d, sampling_frequency: int):
        """Plot the original signal 1D with fft."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot original signal
        axes[0].plot(signal_1d, 'b-', label='Original Signal', alpha=0.5)
        axes[0].plot(envelope_1d, 'r-', label='Envelope')
        axes[0].set_title('Original Signal')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True)
        
        # Calculate and plot FFT of original signal
        fft_signal_1 = np.fft.fft(signal_1d)
        freq_signal_1 = np.fft.fftfreq(len(fft_signal_1), d=1/sampling_frequency)
        axes[1].plot(freq_signal_1[:len(freq_signal_1)//2] / 1e6, np.abs(fft_signal_1)[:len(fft_signal_1)//2], 'b-')
        axes[1].set_title('FFT of Original Signal')
        axes[1].set_xlabel('Frequency [MHz]')
        axes[1].set_ylabel('Magnitude')
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()
    
    ###################################################################################
    # Create color channels 2D
    ###################################################################################
    def create_color_channels_2d(self):
        
        if self.method == "method_1":
            # Implement logic for method_2
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_1(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_2":
            # Implement logic for method_2
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_2(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_3":
            # Implement logic for method_3
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_3(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_4":
            # Implement logic for method_4
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_4(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_5":
            # Implement logic for method_5
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_5(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )
            
        elif self.method == "method_6":
            # Implement logic for method_5
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_6(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )
            
    ###################################################################################   
    # Hscan core method 1
    ###################################################################################   
    def hscan_core_method_1(self,
                            red_xd,
                            green_xd,
                            blue_xd):

        # Normalize envelope for color channel and get log
        red_xd   = self.normalize_with_min_max(red_xd, 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
        
        # Log transformation
        red_xd = np.log10(red_xd)
        green_xd = np.log10(green_xd)
        blue_xd = np.log10(blue_xd)
                        
        # Difference calculation
        RB_difference_xd = red_xd - blue_xd
        positive, negative = self.positive_negative_separator_xd(arr_xd=RB_difference_xd, limit=0)
        
        # Assign values based on the difference
        red_xd = positive
        green_xd = green_xd
        blue_xd = np.abs(negative)
        
        return red_xd, green_xd, blue_xd
        
    ###################################################################################   
    # Hscan core method 2
    def hscan_core_method_2(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # Normalize color channel 0-255
        red_xd   = self.normalize_with_min_max(red_xd , 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
        
        # log
        red_xd   = np.log10(red_xd)
        green_xd = np.log10(green_xd)
        blue_xd  = np.log10(blue_xd)
            
        return red_xd, green_xd, blue_xd
                
    ###################################################################################   
    # Hscan core method 3
    ###################################################################################   
    def hscan_core_method_3(self,
                        red_xd,
                        green_xd,
                        blue_xd):
    
        # Normalize color channel 0-255
        red_xd   = self.normalize_with_min_max(red_xd , 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
                   
        return red_xd, green_xd, blue_xd
    
    ###################################################################################   
    # Hscan core method 4
    ###################################################################################   
    def hscan_core_method_4(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # Normalize color channel 0-255
        red_xd   = self.normalize_with_min_max(red_xd , 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
        
        # log
        red_xd   = np.log10(red_xd)
        green_xd = np.log10(green_xd)
        blue_xd  = np.log10(blue_xd)
            
        # shift to positive
        red_xd   = self.shift_to_positive(red_xd)
        green_xd = self.shift_to_positive(green_xd)
        blue_xd  = self.shift_to_positive(blue_xd)
    
        return red_xd, green_xd, blue_xd    
    
    ###################################################################################   
    # Hscan core method 5
    ###################################################################################   
    def hscan_core_method_5(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # log
        red_xd   = 20 * np.log10(np.abs(1 + red_xd))
        green_xd = 20 * np.log10(np.abs(1 + green_xd))
        blue_xd  = 20 * np.log10(np.abs(1 + blue_xd))
        
        # shift to positive
        red_xd   = self.shift_to_positive(red_xd)
        green_xd = self.shift_to_positive(green_xd)
        blue_xd  = self.shift_to_positive(blue_xd)
        
        # Normalize color channel 0-1
        red_xd   = self.normalize_to_0_1(red_xd)
        green_xd = self.normalize_to_0_1(green_xd)
        blue_xd  = self.normalize_to_0_1(blue_xd)
        
        return red_xd, green_xd, blue_xd
                        
    ###################################################################################   
    # Hscan core method 6
    ###################################################################################   
    def hscan_core_method_6(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # Normalize color channel 0-1
        red_xd   = self.normalize_to_0_1(red_xd)
        green_xd = self.normalize_to_0_1(green_xd)
        blue_xd  = self.normalize_to_0_1(blue_xd)
                    
        return red_xd, green_xd, blue_xd
        
    ###################################################################################
    # Prepare R0B 3x2D
    ###################################################################################
    def prepare_R0B_3x2d(self):
        
        # create R0B channel with mask
        self.R0B_trimmed_3x2d = np.stack([self.R_channel_2d,
                                            np.zeros_like(self.G_channel_2d),
                                            self.B_channel_2d], axis=-1)    
        
    ###################################################################################
    # Prepare RGB 3x2D
    ###################################################################################
    def prepare_RGB_3x2d(self):

        # full RGB image
        self.RGB_trimmed_3x2d = np.stack([self.R_channel_2d,
                                            self.G_channel_2d,
                                            self.B_channel_2d], axis=-1)
    
    ###################################################################################
    # Image envelope
    ###################################################################################
    def image_envelope_2d(self, envelope_2d: np.ndarray, title: str, clip_fact: float, dyn_range: float) -> None:
        """
        Plots a 2D signal envelope in decibels.

        This function takes a 2D array representing a signal envelope, applies 
        a rotation and flipping transformation, converts it to a logarithmic 
        decibel scale, applies clipping and normalization, and displays it as an image plot.

        Args:
            envelope (np.ndarray): The 2D signal envelope array.
            title (str): The title of the plot.
            clip_fact (float): Clipping factor for dynamic range.
            dyn_range (float): Dynamic range in dB.

        Returns:
            None
        """
        
        log_envelope_2d = 20 * np.log10(np.abs(1 + envelope_2d))

        # Step 2: Clip and normalize
        clipped_max = clip_fact * np.amax(log_envelope_2d)
        log_envelope_2d = np.clip(log_envelope_2d, clipped_max - dyn_range, clipped_max)
        log_envelope_2d -= np.amin(log_envelope_2d)
        log_envelope_2d *= (255 / np.amax(log_envelope_2d))

        plt.figure(figsize=(8, 6))
        plt.imshow(log_envelope_2d, cmap='gray', aspect='auto')
        plt.title(title)
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    ###################################################################################
    # Image RB difference in RGB
    ###################################################################################
    def image_RB_difference_in_RGB_2d(self,
                                        R0B_final_2d: np.ndarray,  # RB final image as a numpy array
                                        additional_text: str = "",
                                        log: bool = False,  # Set default to False
                                        ) -> None:  # Default filename

        # Define custom color points for the colormap
        colors = [(1, 0, 0), (0, 0, 0), (0, 0, 1)]  # Red, Black, Blue
        
        # Create a custom colormap
        cmap_name = 'my_custom_colorbar'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        
        # Create a new figure
        plt.figure(figsize=(8, 6))
        
        if log:
            R0B_final_2d = 20 * np.log10(np.abs(1 + R0B_final_2d))

        R0B_final_2d = self.rotate_flip(R0B_final_2d)
        
        # Display the image
        plt.imshow(R0B_final_2d, aspect='auto', cmap=custom_cmap)

        plt.title(f'RB image zero G in RGB, {additional_text}')
        
        # Add a color bar with dynamic ticks
        cbar = plt.colorbar()
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Large', 'Medium', 'Small'])

        # Adjust layout for better display
        plt.tight_layout()
        plt.show()
  
    ###################################################################################
    # Image color channel 2D
    ###################################################################################
    def image_color_channel_2d(self,
                                color_channel_2d: np.ndarray,   # Red channel without RGB as a numpy array
                                color = None,
                                addtional_text = None,
                                log: bool = None, 
                                ) -> None:         # Flag indicating whether to save the image

        # Create a new figure
        plt.figure(figsize=(8, 6))
        
        if log:
            
            color_channel_2d = 20 * np.log10(np.abs(1 + color_channel_2d))

            if color == "red":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Reds')
            elif color == "green":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Greens')
            elif color == "blue":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Blues')
            elif color == "RGB":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto')
                #plt.colorbar().remove()
        
        else:    
            
            if color == "red":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Reds')
            elif color == "green":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Greens')
            elif color == "blue":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Blues')
            elif color == "RGB":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto')
                #plt.colorbar().remove()
        
        # Add a color bar
        plt.colorbar()

        # Set the title of the plot
        plt.title(f'{addtional_text}')  
        
        # Adjust layout for better display
        plt.tight_layout()

        plt.show()

    ###################################################################################
    # Plot 1D signal envelope and fft
    ###################################################################################
    def plot_1d_signal_envelope_and_fft(self, signal, envelope):
        # Generate time axis based on the length of the signal
        num_samples = len(signal)
        time_microseconds = np.arange(num_samples)  # Use sample indices as time in microseconds

        # Calculate FFT
        fft_signal = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_signal)

        # Frequency axis in MHz using the sampling frequency
        freq_MHz = np.fft.fftfreq(num_samples, d=(1 / (self.sampling_rate_MHz)))  # Frequency in Hz

        # Only take positive frequencies
        positive_freq_indices = freq_MHz >= 0
        freq_positive = freq_MHz[positive_freq_indices]
        fft_magnitude_positive = fft_magnitude[positive_freq_indices]

        # Create figure
        plt.figure(figsize=(14, 4))

        # Plot the original signal and its envelope
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.plot(time_microseconds, signal, color='blue', label='Signal')  # Original signal
        plt.plot(time_microseconds, envelope, color='orange', linestyle='--', label='Envelope')  # Envelope
        plt.title('1D Signal Plot')
        plt.xlabel('samples')  # Label for microseconds
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # Plot the positive FFT
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.plot(freq_positive, fft_magnitude_positive, color='green')  # Frequency in MHz
        plt.title('FFT of Signal')
        plt.xlabel('Frequency (MHz)')  # Label for MHz
        plt.ylabel('Magnitude')
        plt.grid(True)

        plt.tight_layout()  # Adjust layout
        plt.show()
        
    ###################################################################################
    # Plot envelope
    ###################################################################################
    def plot_envelope(self, envelope, color):
        # Generate time axis based on the length of the envelope
        num_samples = len(envelope)
        time_microseconds = np.arange(num_samples)  # Use sample indices as time in microseconds

        # Create figure
        plt.figure(figsize=(7, 4))

        # Plot the envelope with the specified color
        plt.plot(time_microseconds, envelope, color=color, linestyle='--', label='Envelope')  # Envelope
        plt.title('Envelope of the Signal')
        plt.xlabel('Samples')  # Label for microseconds
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()  # Adjust layout
        plt.show()

    

    
       
    
    
###################################################################################
# BSC post-processing
###################################################################################
class BSCPostProcessing:
    """Class for post-processing the BSC data."""

    def __init__(self):
        pass

###################################################################################
# Central frequency shift post-processing
###################################################################################
class CentralFrequencyShiftPostProcessing:
    """Class for post-processing the central frequency shift data."""

    def __init__(self):
        pass

###################################################################################
# 
###################################################################################
    
