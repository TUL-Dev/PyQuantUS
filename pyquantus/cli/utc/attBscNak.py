import numpy as np
from typing import Tuple
from scipy.signal import hilbert

from pyquantus.utc.objects import Window
from pyquantus.utc.analysis import UtcAnalysis, UtcResults
from pyquantus.utc.transforms import computeHanningPowerSpec

class AbnResults(UtcResults):
    """Add attenuation coefficient, backscatter coefficient, and Nakagami parameters as results
    for each window to store when completing UTC analysis.
    """
    def __init__(self):
        super().__init__()
        self.attCoef: float  # attenuation coefficient (dB/cm/MHz)
        self.bsc: float  # backscatter coefficient (1/cm-sr)
        self.uNakagami: float  # Nakagami shape parameter
        self.wNakagami: float  # Nakagami scale parameter
    
# Specify format for analysis kwargs, which are new parameters
# needed for this analysis not already in the UtcAnalysis class.
analysisKwargs = {
    'refAttenuation': float,  # dB/cm/MHz
    'refBackScatterCoef': float,  # 1/cm-sr
} # REQUIRED

class AbnAnalysis(UtcAnalysis):
    """Class to compute the attenuation coefficient, backscatter coefficient, and Nakagami parameters
    on RF data as well as the base UTC parameters (MBF, SS, SI).
    """
    def __init__(self, kwargTypes, **kwargs):
        super().__init__(kwargTypes, **kwargs)
        self.ResultsClass = AbnResults
        
        # New parameters for the analysis - expecting refAttenuation and refBackScatterCoef
        self.__dict__.update(kwargs)
        
    ################# OVERRIDDEN FUNCTIONS #################
        
    def computeParamapVals(self, scanRfWindow: np.ndarray, phantomRfWindow: np.ndarray, window: Window):
        """Modify the computations completed on each window when generating a parametric map.
        Note if these parameters are not wanted for a parameteric map, this step can be skipped.
        """
        # Computes MBF, SS, SI and also stores NPS, PS, rPS, and f in window.results
        self.computeBaseParamapVals(scanRfWindow, phantomRfWindow, window)
            
        attCoef = self.computeAttenuationCoef(scanRfWindow, phantomRfWindow, windowDepth=min(100, scanRfWindow.shape[0]//5))
        bsc = self.computeBackscatterCoefficient(window.results.f, window.results.ps, window.results.rPs, 
                                                    attCoef, self.config.centerFrequency, roiDepth=scanRfWindow.shape[0])
        _, uNakagami = self.computeNakagamiParams(scanRfWindow)
        window.results.attCoef = attCoef # dB/cm/MHz
        window.results.bsc = bsc # 1/cm-sr
        window.results.uNakagami = uNakagami
        
    def computeSingleWindowVals(self, scanRfWindow, phantomRfWindow):
        """Modify the computations completed on a single window containing the entire ROI
        to include the attenuation coefficient, backscatter coefficient, and Nakagami parameters.
        Note if these parameters are not wanted for this large single window, 
        this step can be skipped.
        """
        self.computeBaseParamapVals(scanRfWindow, phantomRfWindow, self.singleWindow)
        
        attCoef = self.computeAttenuationCoef(scanRfWindow, phantomRfWindow, windowDepth=min(100, scanRfWindow.shape[0]//5))
        bsc = self.computeBackscatterCoefficient(self.singleWindow.results.f, self.singleWindow.results.ps, 
                                                 self.singleWindow.results.rPs, attCoef, 
                                                 self.config.centerFrequency, roiDepth=scanRfWindow.shape[0])
        wNakagami, uNakagami = self.computeNakagamiParams(scanRfWindow)
        
        self.singleWindow.results.attCoef = attCoef
        self.singleWindow.results.bsc = bsc
        self.singleWindow.results.uNakagami = uNakagami
        self.singleWindow.results.wNakagami = wNakagami
        
    ################# CUSTOM FUNCTIONS #################        
        
    def computeAttenuationCoef(self, rfData: np.ndarray, refRfData: np.ndarray, overlap=50, windowDepth=100) -> float:
        """Compute the local attenuation coefficient of the ROI using the Spectral Difference
        Method for Local Attenuation Estimation. This method computes the attenuation coefficient
        for multiple frequencies and returns the slope of the attenuation as a function of frequency.
        Args:
            rfData (np.ndarray): RF data of the ROI (n lines x m samples).
            refRfData (np.ndarray): RF data of the phantom (n lines x m samples).
            overlap (float): Overlap percentage for analysis windows.
            windowDepth (int): Depth of each window in samples.
        Returns:
            float: Local attenuation coefficient of the ROI for the central frequency (dB/cm/MHz).
            Updated and verified : Feb 2025 - IR
        """
        samplingFrequency = self.config.samplingFrequency
        startFrequency = self.config.analysisFreqBand[0]
        endFrequency = self.config.analysisFreqBand[1]
        # Initialize arrays for storing intensities (log of power spectrum for each frequency)
        psSample = [];  # ROI power spectra
        psRef = [];  # Phantom power spectra
        startIdx = 0
        endIdx = windowDepth
        windowCenterIndices = []
        counter = 0
        # Loop through the windows in the RF data
        while endIdx < rfData.shape[0]:
            subWindowRf = rfData[startIdx: endIdx]
            f, ps = computeHanningPowerSpec(subWindowRf, startFrequency, endFrequency, samplingFrequency)
            psSample.append(20*np.log10(ps))  # Log scale intensity for the ROI
            refSubWindowRf = refRfData[startIdx: endIdx]
            refF, refPs = computeHanningPowerSpec(refSubWindowRf, startFrequency, endFrequency, samplingFrequency)
            psRef.append(20*np.log10(refPs))  # Log scale intensity for the phantom
            windowCenterIndices.append((startIdx + endIdx) // 2)
            startIdx += int(windowDepth*(1-(overlap/100)))
            endIdx = startIdx + windowDepth
            counter += 1
        # Convert window depths to cm
        axialResCm = self.ultrasoundImage.axialResRf / 10
        windowDepthsCm = np.array(windowCenterIndices) * axialResCm
        attenuationCoefficients = []  # One coefficient for each frequency
        f = f / 1e6
        psSample = np.array(psSample)
        psRef = np.array(psRef)
        midIdx = f.shape[0] // 2  # Middle index
        startIdx = max(0, midIdx - 25)  # Start index for slicing
        endIdx = min(f.shape[0], midIdx + 25)  # End index for slicing
        # Compute attenuation for each frequency
        for fIdx in range(startIdx, endIdx):
            normalizedIntensities = np.subtract(psSample[:, fIdx], psRef[:, fIdx])
            p = np.polyfit(windowDepthsCm, normalizedIntensities, 1)
            localAttenuation = self.refAttenuation * f[fIdx] - (1 / 4) * p[0]  # dB/cm
            attenuationCoefficients.append( localAttenuation / f[fIdx])  # dB/cm/MHz
        attenuationCoef=np.mean(attenuationCoefficients)
        return attenuationCoef
    
    def computeBackscatterCoefficient(self, freqArr: np.ndarray, scanPs: np.ndarray, refPs: np.ndarray, attCoef: float,
                                      frequency: int, roiDepth: int) -> float:
        
        """Compute the backscatter coefficient of the ROI using the reference phantom method.
        Assumes instrumentation and beam terms have the same effect on the signal from both 
        image and phantom. 
        source: Yao et al. (1990) : https://doi.org/10.1177/016173469001200105. PMID: 2184569
        Args:
            freqArr (np.ndarray): Frequency array of power spectra (Hz).
            scanPs (np.ndarray): Power spectrum of the analyzed scan at the current region.
            refPs (np.ndarray): Power spectrum of the reference phantom at the currentn region.
            attCoef (float): Attenuation coefficient of the current region (dB/cm/MHz).
            frequency (int): Frequency on which to compute backscatter coefficient (should 
                    match frequency of self.refBackScatterCoefficient) (MHz).
            roiDepth (int): Depth of the start of the ROI in samples.
            
        Returns:
            float: Backscatter coefficient of the ROI for the central frequency (1/cm-sr).
            Updated and verified : Feb 2025 - IR
        """
        index = np.argmin(np.abs(freqArr - frequency))
        psSample=(scanPs[index])
        psRef=(refPs[index])
        sRatio = psSample/psRef
        
        npConversionFactor = np.log(10) / 20 
        convertedAttCoef = attCoef * npConversionFactor  # dB/cm/MHz -> Np/cm/MHz
        convertedRefAttCoef = self.refAttenuation * npConversionFactor # dB/cm/MHz -> Np/cm/MHz
        windowDepthCm = roiDepth*self.ultrasoundImage.axialResRf/10 # cm
        
        attComp=np.exp(4*windowDepthCm *(convertedAttCoef-convertedRefAttCoef)) 
        bsc = sRatio*self.refBackScatterCoef*attComp
            
        return bsc
        
    def computeNakagamiParams(self, rfData: np.ndarray) -> Tuple[float, float]:
        """Compute Nakagami parameters for the ROI.
        source: Tsui, P. H., Wan, Y. L., Huang, C. C. & Wang, M. C. 
        Effect of adaptive threshold filtering on ultrasonic Nakagami 
        parameter to detect variation in scatterer concentration. Ultrason. 
        Imaging 32, 229–242 (2010). https://doi.org/10.1177%2F016173461003200403

        Args:
            rfData (np.ndarray): RF data of the ROI (n lines x m samples).
            
        Returns:
            Tuple: Nakagami parameters (w, u) for the ROI.
        """
        
        r = np.abs(hilbert(rfData, axis=1))
        w = np.nanmean(r ** 2, axis=1)
        u = (w ** 2) / np.var(r ** 2, axis=1)

        # Added this to get single param values
        w = np.nanmean(w)
        u = np.nanmean(u)

        return w, u
