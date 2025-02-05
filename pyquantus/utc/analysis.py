from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import hilbert
from scipy.optimize import curve_fit

from pyquantus.utc.objects import UltrasoundImage, AnalysisConfig, Window
from pyquantus.utc.transforms import computeHanningPowerSpec, computeSpectralParams

class UtcAnalysis:
    """Complete ultrasound tissue characterization (UTC) analysis of an ultrasound image given 
    a corresponding phantom image.

    This class supports both scan converted and non-scan converted images. UTC analysis
    is performed on each window generated by the `generateRoiWindows` method. Utc 
    analysis corresponds to the computation of tissue characterization-related parameters.
    The midband fit, spectral slope, and spectral intercept parameters have been validated using
    the frequency domain of each window. The Nakagami parameters, attenuation coefficient,
    backscatter coefficient, effective scatterer diameter, and effective scatterer concentration
    all have been implemented and reviewed, but still have not been validated in practice.

    Attributes:
        ultrasoundImage (UltrasoundImage): Stores image and RF info for image and phantom.
        config (AnalysisConfig): Stores analysis configuration parameters.
        roiWindows (List[Window]): List of windows generated by `generateRoiWindows`.
        waveLength (float): Wavelength of the ultrasound signal in mm.
        nakagamiParams (Tuple): Nakagami parameters (w, u) for the entire ROI.
        attenuationCoef (float): Attenuation coefficient of the entire ROI at the center frequency (dB/cm/MHz).
        backScatterCoef (float): Backscatter coefficient of the entire ROI at the center frequency (1/cm-sr).
        effectiveScattererDiameter (float): Effective scatterer diameter of the entire ROI (µm).
        effectiveScattererConcentration (float): Effective scatterer concentration of the entire ROI (dB/mm^3).
        refAttenuation (float): Total attenuation coefficient of the reference phantom at the center frequency (dB/cm/MHz).
        refBackScatterCoef (float): Backscatter coefficient of the reference phantom at the center frequency (1/cm-sr).
        scSplineX (np.ndarray): Spline x-coordinates in scan converted coordinates.
        splineX (np.ndarray): Spline x-coordinates in pre-scan converted coordinates.
        scSplineY (np.ndarray): Spline y-coordinates in scan converted coordinates.
        splineY (np.ndarray): Spline y-coordinates in pre-scan converted coordinates.
    """
    
    def __init__(self):
        self.ultrasoundImage: UltrasoundImage
        self.config: AnalysisConfig
        self.roiWindows: List[Window] = []
        self.waveLength: float
        self.nakagamiParams: Tuple
        self.attenuationCoef: float
        self.refBackScatterCoef: float
        self.effectiveScattererDiameter: float
        self.effectiveScattererConcentration: float
        self.refAttenuation: float
        self.backScatterCoef: float

        self.scSplineX: np.ndarray # pix
        self.splineX: np.ndarray # pix
        self.scSplineY: np.ndarray # pix
        self.splineY: np.ndarray # pix

    def initAnalysisConfig(self):
        """Compute the wavelength of the ultrasound signal and 
        set default config values if not pre-loaded.
        """
        speedOfSoundInTissue = 1540  # m/s
        self.waveLength = (
            speedOfSoundInTissue / self.config.centerFrequency
        ) * 1000  # mm
        if not hasattr(self.config, 'axWinSize'): # not pre-loaded config
            self.config.axWinSize = 10 * self.waveLength
            self.config.latWinSize = 10 * self.waveLength
            self.config.axialOverlap = 0.5; self.config.lateralOverlap = 0.5
            self.config.windowThresh = 0.95

    def splineToPreSc(self):
        """Convert spline coordinates from scan converted to pre-scan converted."""
        self.splineX = np.array([self.ultrasoundImage.xmap[int(y), int(x)] for x, y in zip(self.scSplineX, self.scSplineY)])
        self.splineY = np.array([self.ultrasoundImage.ymap[int(y), int(x)] for x, y in zip(self.scSplineX, self.scSplineY)])

    def generateRoiWindows(self):
        """Generate windows for UTC analysis based on user-defined spline."""
        # Some axial/lateral dims
        axialPixSize = round(self.config.axWinSize / self.ultrasoundImage.axialResRf) # mm/(mm/pix)
        lateralPixSize = round(self.config.latWinSize / self.ultrasoundImage.lateralResRf) # mm(mm/pix)
        axial = list(range(self.ultrasoundImage.rf.shape[0]))
        lateral = list(range(self.ultrasoundImage.rf.shape[1]))

        # Overlap fraction determines the incremental distance between ROIs
        axialIncrement = axialPixSize * (1 - self.config.axialOverlap)
        lateralIncrement = lateralPixSize * (1 - self.config.lateralOverlap)

        # Determine ROIS - Find Region to Iterate Over
        axialStart = max(min(self.splineY), axial[0])
        axialEnd = min(max(self.splineY), axial[-1] - axialPixSize)
        lateralStart = max(min(self.splineX), lateral[0])
        lateralEnd = min(max(self.splineX), lateral[-1] - lateralPixSize)

        self.roiWindows = []

        # Determine all points inside the user-defined polygon that defines analysis region
        # The 'mask' matrix - "1" inside region and "0" outside region
        # Pair x and y spline coordinates
        spline = []
        if len(self.splineX) != len(self.splineY):
            print("Spline has unequal amount of x and y coordinates")
            return
        for i in range(len(self.splineX)):
            spline.append((self.splineX[i], self.splineY[i]))

        img = Image.new("L", (self.ultrasoundImage.rf.shape[1], self.ultrasoundImage.rf.shape[0]), 0)
        ImageDraw.Draw(img).polygon(spline, outline=1, fill=1)
        mask = np.array(img)

        for axialPos in np.arange(axialStart, axialEnd, axialIncrement):
            for lateralPos in np.arange(lateralStart, lateralEnd, lateralIncrement):
                # Convert axial and lateral positions in mm to Indices
                axialAbsAr = abs(axial - axialPos)
                axialInd = np.where(axialAbsAr == min(axialAbsAr))[0][0]
                lateralAbsAr = abs(lateral - lateralPos)
                lateralInd = np.where(lateralAbsAr == min(lateralAbsAr))[0][0]

                # Determine if ROI is Inside Analysis Region
                maskVals = mask[
                    axialInd : (axialInd + axialPixSize),
                    lateralInd : (lateralInd + lateralPixSize),
                ]

                # Define Percentage Threshold
                totalNumberOfElementsInRegion = maskVals.size
                numberOfOnesInRegion = len(np.where(maskVals == 1)[0])
                percentageOnes = numberOfOnesInRegion / totalNumberOfElementsInRegion

                if percentageOnes > self.config.windowThresh:
                    # Add ROI to output structure, quantize back to valid distances
                    newWindow = Window()
                    newWindow.left = int(lateral[lateralInd])
                    newWindow.right = int(lateral[lateralInd + lateralPixSize - 1])
                    newWindow.top = int(axial[axialInd])
                    newWindow.bottom = int(axial[axialInd + axialPixSize - 1])
                    self.roiWindows.append(newWindow)
    
    def computeUtcWindows(self, extraParams=True, bscFreq=None) -> int:
        """Compute UTC parameters for each window in the ROI.
        
        extraParams (bool): Flag on whether to compute non-validated parameters.
        bscFreq (int): Frequency on which to compute backscatter coefficient (MHz).
        
        Returns:
            int: 0 if successful, -1 if `generateRoiWindows` has not been 
            run or if windows are too large for ROI.
        """
        if not len(self.roiWindows):
            print("Run 'generateRoiWindows' first")
            return -1
        
        if bscFreq is None:
            bscFreq = self.config.centerFrequency
    
        fs = self.config.samplingFrequency
        f0 = self.config.transducerFreqBand[0]
        f1 = self.config.transducerFreqBand[1]
        lowFreq = self.config.analysisFreqBand[0]
        upFreq = self.config.analysisFreqBand[1]

        # Compute MBF, SS, and SI parameters for each window
        for window in self.roiWindows:
            # Compute normalized power spectrum (dB)
            imgWindow = self.ultrasoundImage.rf[window.top: window.bottom+1, window.left: window.right+1]
            refWindow = self.ultrasoundImage.phantomRf[window.top: window.bottom+1, window.left: window.right+1]
            f, ps = computeHanningPowerSpec(
                imgWindow, f0, f1, fs
            ) 
            ps = 20 * np.log10(ps)
            f, rPs = computeHanningPowerSpec(
                refWindow, f0, f1, fs
            )
            rPs = 20 * np.log10(rPs)
            nps = np.asarray(ps) - np.asarray(rPs)

            window.results.nps = nps
            window.results.ps = np.asarray(ps)
            window.results.rPs = np.asarray(rPs)
            window.results.f = np.asarray(f)

            # Compute MBF, SS, and SI
            mbf, _, _, p = computeSpectralParams(nps, f, lowFreq, upFreq)
            window.results.mbf = mbf # dB
            window.results.ss = p[0]*1e6 # dB/MHz
            window.results.si = p[1] # dB

        minLeft = min([window.left for window in self.roiWindows])
        maxRight = max([window.right for window in self.roiWindows])
        minTop = min([window.top for window in self.roiWindows])
        maxBottom = max([window.bottom for window in self.roiWindows])

        if extraParams:
            imgWindow = self.ultrasoundImage.rf[minTop: maxBottom+1, minLeft: maxRight+1]
            refWindow = self.ultrasoundImage.phantomRf[minTop: maxBottom+1, minLeft: maxRight+1]
            self.attenuationCoef = self.computeAttenuationCoef(imgWindow, refWindow)
            self.backScatterCoef = self.computeBackscatterCoefficient(imgWindow, refWindow, bscFreq, roiDepth=minTop)
            self.nakagamiParams = self.computeNakagamiParams(imgWindow) # computing for entire ROI, but could also be easily computed for each window
            self.effectiveScattererDiameter, self.effectiveScattererConcentration = self.computeEsdac(imgWindow, refWindow, apertureRadiusCm=6)
        
        return 0
    
    def computeAttenuationCoef(self, rfData: np.ndarray, refRfData: np.ndarray,
                                    overlap=50, windowDepth=100) -> float:
        """Compute the local attenuation coefficient of the ROI using the Spectral Difference
        Method for Local Attenuation Estimation. Note this method cannot be used to estimate
        attenuation when the scattering properties change within the ROI. This method assumes
        a tissue-mimicking phantom with a sound speed similar to the expected sound speed of 
        the analyzed tissue. Assuming linear dependency on depth, the local attenuation coefficient
        computed here is equivalent to the total attenuation coefficient.
        source: Mamou & Oelze, page 78-79: https://doi.org/10.1007/978-94-007-6952-6
        
        Args:
            rfData (np.ndarray): RF data of the ROI (n lines x m samples).
            refRfData (np.ndarray): RF data of the phantom (n lines x m samples).
            roiDepth (int): Depth of the start of the ROI in samples.
            overlap (float): Overlap percentage for analysis windows.
            windowDepth (int): Depth of each window in samples.
        
        Returns:
            float: Local attenuation coefficient of the ROI for the central frequency (dB/cm/MHz).
        """
        samplingFrequency = self.config.samplingFrequency
        startFrequency = self.config.analysisFreqBand[0]
        endFrequency = self.config.analysisFreqBand[1]

        intensities = []; refIntensities = []
        startIdx = 0; endIdx = windowDepth; windowCenterIndices = []
        while endIdx < rfData.shape[0]:
            subWindowRf = rfData[startIdx: endIdx]
            f, ps = computeHanningPowerSpec(subWindowRf, startFrequency, endFrequency, samplingFrequency)
            ps = np.log(ps)
            intensities.append(ps[len(f)//2])
            
            refSubWindowRf = refRfData[startIdx: endIdx]
            refF, refPs = computeHanningPowerSpec(refSubWindowRf, startFrequency, endFrequency, samplingFrequency)
            refPs = np.log(refPs)
            refIntensities.append(refPs[len(refF)//2])
            
            windowCenterIndices.append((startIdx+endIdx)//2)
            startIdx += int(windowDepth*(overlap/100))
            endIdx = startIdx + windowDepth

        axialResCm = self.ultrasoundImage.axialResRf/10
        windowDepthsCm = np.array(windowCenterIndices)*axialResCm # sample * (cm/sample) = cm
        normalizedIntensities = np.subtract(intensities, refIntensities)
        p = np.polyfit(windowDepthsCm, normalizedIntensities, 1)
        localAttenuation = self.refAttenuation * self.config.centerFrequency/1e6 - (1/4)*p[0] # dB/cm
        attenuationCoef = localAttenuation /  (self.config.centerFrequency/1e6) # dB/cm/MHz
        
        return attenuationCoef
    
    # def computeTotalAttenuationCoef(self, rfData: np.ndarray, refRfData: np.ndarray, roiDepth: int, 
    #                                 overlap=50, windowDepth=100) -> float:
    #     """Compute the total attenuation coefficient of the ROI using the Spectral Fit Algorithm
    #     for Total Attenuation Estimation. This method assumes a Gaussian form factor for both the image
    #     and phantom scans. It also assumes total attenuation for the image has a linear frequency dependence.
    #     source: Mamou & Oelze, page 88-89: https://doi.org/10.1007/978-94-007-6952-6
        
    #     Args:
    #         rfData (np.ndarray): RF data of the ROI (n lines x m samples).
    #         refRfData (np.ndarray): RF data of the phantom (n lines x m samples).
    #         roiDepth (int): Depth of the start of the ROI in samples.
    #         overlap (float): Overlap percentage for analysis windows.
    #         windowDepth (int): Depth of each window in samples.
        
    #     Returns:
    #         float: Total attenuation coefficient of the ROI for the central frequency.
    #     """
    #     samplingFrequency = self.config.samplingFrequency
    #     startFrequency = self.config.analysisFreqBand[0]
    #     endFrequency = self.config.analysisFreqBand[1]
        
    #     intensities = []; refIntensities = []
    #     startIdx = 0; endIdx = windowDepth; windowCenterIndices = []
    #     while endIdx > rfData.shape[0]:
    #         subWindowRf = rfData[startIdx: endIdx]
    #         f, ps = computeHanningPowerSpec(subWindowRf, startFrequency, endFrequency, samplingFrequency)
    #         intensities.append(ps[len(f)//2])
            
    #         refSubWindowRf = refRfData[startIdx: endIdx]
    #         refF, refPs = computeHanningPowerSpec(refSubWindowRf, startFrequency, endFrequency, samplingFrequency)
    #         refIntensities.append(refPs[len(refF)//2])
            
    #         windowCenterIndices.append((startIdx+endIdx)//2)
    #         startIdx += int(windowDepth*(overlap/100))
    #         endIdx = startIdx + windowDepth
        
    #     axialResCm = self.ultrasoundImage.axialResRf/10
    #     zT = axialResCm*(windowCenterIndices+roiDepth) # cm
    #     sFit = (intensities/refIntensities)*np.exp(-self.refAttenuation*zT)*self.refBackScatterCoef
    #     f *= 1e-6 # MHz
    #     tissueSpeedOfSound = 154000 # cm/s
        
    #     def fitFunc(attenuationSlope, effectiveScattererRadius, const):
    #         return 0.827*(2*np.pi*effectiveScattererRadius/tissueSpeedOfSound)*(f**2) + 4*attenuationSlope*f*zT + const
        
    #     # Fit the attenuation slope and effective scatterer radius
    #     # Use scipy.optimize.curve_fit
    #     popt, _ = curve_fit(fitFunc, [f], [sFit])
    #     attenuationSlope = popt[0]
    #     totalAttenuation = attenuationSlope*f[len(f)//2]
        
    #     return totalAttenuation
    
    def computeBackscatterCoefficient(self, rfData: np.ndarray, refRfData: np.ndarray,
                                      frequency: int, roiDepth: int) -> float:
        """Compute the backscatter coefficient of the ROI using the reference phantom method.
        Assumes instrumentation and beam terms have the same effect on the signal from both 
        image and phantom. 
        source: Mamou & Oelze, page 52: https://doi.org/10.1007/978-94-007-6952-6

        Args:
            rfData (np.ndarray): RF data of the ROI (n lines x m samples).
            refRfData (np.ndarray): RF data of the phantom (n lines x m samples).
            roiDepth (int): Depth of the start of the ROI in samples.
            frequency (int): Frequency on which to compute backscatter coefficient (MHz).
            refAttenuation (float): Attenuation coefficient of the reference phantom.
            
        Returns:
            float: Backscatter coefficient of the ROI for the central frequency (1/cm-sr).
        """
        f, ps = computeHanningPowerSpec(rfData, self.config.analysisFreqBand[0], self.config.analysisFreqBand[1],
                                     self.config.samplingFrequency)
        _, refPs = computeHanningPowerSpec(refRfData, self.config.analysisFreqBand[0], self.config.analysisFreqBand[1],
                                     self.config.samplingFrequency)
        
        ps = ps[len(f)//2]
        refPs = refPs[len(f)//2]
        axialResM = self.ultrasoundImage.axialResRf/1000 # m
        depthDistance = roiDepth*axialResM + 0.5*rfData.shape[0]*axialResM # m, center of ROI
        
        npConversionFactor = np.log(10) / 20 
        convertedAttCoef = self.attenuationCoef * npConversionFactor / 100 # dB/cm/MHz -> Np/m/MHz
        convertedRefAttCoef = self.refAttenuation * npConversionFactor / 100 # dB/cm/MHz -> Np/m/MHz
        convertedAttCoef *= frequency # Np/m
        convertedRefAttCoef *= frequency # Np/m
        
        bsc = (ps/refPs)*self.refBackScatterCoef*np.exp(4*depthDistance*(convertedAttCoef - convertedRefAttCoef)) # 1/cm/sr
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
    
    def computeEsdac(self, rfData: np.ndarray, refRfData: np.ndarray, apertureRadiusCm: float) -> Tuple[float, float]:
        """Compute the effective scatterer diameter and concentration of the ROI.
        source: Muleki-Seya et al. https://doi.org/10.1177/0161734617729159
        
        Args:
            rfData (np.ndarray): RF data of the ROI (n lines x m samples).
            refRfData (np.ndarray): RF data of the phantom (n lines x m samples).
            apertureRadiusCm (float): Aperture radius in cm.
            roiDepth (int): Depth of the start of the ROI in samples.
            
        Returns:
            Tuple: Effective scatterer diameter (µm) and concentration of the ROI (dB/mm^3).
        """
        windowDepthCm = rfData.shape[0]*self.ultrasoundImage.axialResRf/10 # cm
        windowLengthCm = rfData.shape[1]*self.ultrasoundImage.lateralResRf/10 # cm. Assuming this unit, but not explicitly stated in paper
        q = apertureRadiusCm / windowDepthCm

        f, ps = computeHanningPowerSpec(rfData, self.config.analysisFreqBand[0], self.config.analysisFreqBand[1],
                                     self.config.samplingFrequency)
        _, refPs = computeHanningPowerSpec(refRfData, self.config.analysisFreqBand[0], self.config.analysisFreqBand[1],
                                     self.config.samplingFrequency)
        
        ps = 10*np.log10(ps) # dB
        refPs = 10*np.log10(refPs) # dB
        
        f *= 1e-6 # Hz -> MHz
        s = np.subtract(ps-refPs, 10*np.log10(f**4))
        p = np.polyfit(f**2, s, 1)
        m = abs(p[0])
        b1 = p[1]
        esd = 2*(m/((11.6*(q**2))+52.8)) ** 0.5 # µm

        b0 = self.attenuationCoef
        eac = 64 * (
            (10 ** ((b1 + 2*windowDepthCm*b0) / 10)) 
            / (185 * windowLengthCm * (q**2) * esd**6)
        ) # dB/mm^3

        return esd, eac