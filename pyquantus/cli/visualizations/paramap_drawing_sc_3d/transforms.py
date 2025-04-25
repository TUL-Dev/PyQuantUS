import math
import numpy as np
from typing import Tuple
from scipy.interpolate import interpn

from ...scan_loaders._obj.us_rf_3d import ScUltrasoundRfImage3d

def int32torgb(color):
    """Convert int32 to rgb tuple"""
    rgb = []
    for _ in range(3):
        rgb.append(color&0xff)
        color = color >> 8
    return rgb

def condenseArr(image: np.ndarray) -> np.ndarray:
    """Condense (M,N,3) arr to (M,N) with uint32 data to preserve info"""
    assert len(image.shape) == 3
    assert image.shape[-1] == 3
    
    return np.dstack((image,np.zeros(image.shape[:2], 'uint8'))).view('uint32').squeeze(-1)

def expandArr(image: np.ndarray) -> np.ndarray:
    """Inverse of condenseArr"""
    assert len(image.shape) == 2
    
    fullArr = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            fullArr[i,j] = int32torgb(image[i,j])

    return fullArr.astype('uint8')

def scanConvert3Va(rxLines, lineAngles, planeAngles, beamDist, imgSize, fovSize, z0, interp='linear'):
    pixSizeX = 1/(imgSize[0]-1)
    pixSizeY = 1/(imgSize[1]-1)
    pixSizeZ = 1/(imgSize[2]-1)

    # Create Cartesian grid and convert to polar coordinates
    xLoc = (np.arange(0,1+(pixSizeX/2),pixSizeX)-0.5)*fovSize[0]
    yLoc = (np.arange(0,1+(pixSizeY/2),pixSizeY)-0.5)*fovSize[1]
    zLoc = np.arange(0,1+(pixSizeZ/2),pixSizeZ)*fovSize[2]
    Z, X, Y = np.meshgrid(zLoc, xLoc, yLoc, indexing='ij')

    PHI = np.arctan2(Y, Z+z0)
    TH = np.arctan2(X, np.sqrt(np.square(Y)+np.square(Z+z0)))
    R = np.sqrt(np.square(X)+np.square(Y)+np.square(Z+z0))*(1-z0/np.sqrt(np.square(Y)+np.square(Z+z0)))
    
    zCoords = np.arange(rxLines.shape[0])
    xCoords = np.arange(rxLines.shape[1])   
    yCoords = np.arange(rxLines.shape[2])
    Zcoords, Xcoords, Ycoords = np.meshgrid(zCoords, xCoords, yCoords, indexing='ij')
    coords = Zcoords*rxLines.shape[1]*rxLines.shape[2] + Xcoords*rxLines.shape[2] + Ycoords

    radLineAngles = np.pi*lineAngles/180
    radPlaneAngles = np.pi*planeAngles/180
    
    img = interpn((beamDist, radLineAngles, radPlaneAngles), 
                                    rxLines, (R, TH, PHI), method=interp, bounds_error=False, fill_value=0)
    
    imgCoords = interpn((beamDist, radLineAngles, radPlaneAngles), 
                                    coords, (R, TH, PHI), method='nearest', bounds_error=False, fill_value=-1)
    
    img = np.array(img)
    imgCoords = np.array(imgCoords, dtype=np.int64)

    return img, imgCoords

def scanConvert3dVolumeSeries(dbEnvDatFullVolSeries, image_data: ScUltrasoundRfImage3d, isLin=True, scale=True, interp='linear', normalize=True) -> Tuple[np.ndarray, np.ndarray, list]:
    if len(dbEnvDatFullVolSeries.shape) != 4:
        numVolumes = 1
        nz, nx, ny = dbEnvDatFullVolSeries.shape
    else:
        numVolumes = dbEnvDatFullVolSeries.shape[0]
        nz, nx, ny = dbEnvDatFullVolSeries[0].shape
    apexDist = image_data.apex_dist
    azimSteerAngleStart = image_data.azim_steer_angle_start
    azimSteerAngleEnd = image_data.azim_steer_angle_end
    rxAngAz = np.linspace(azimSteerAngleStart, azimSteerAngleEnd, nx) # Steering angles in degree
    elevSteerAngleStart = image_data.elev_steer_angle_start
    elevSteerAngleEnd = image_data.elev_steer_angle_end
    rxAngEl = np.linspace(elevSteerAngleStart, elevSteerAngleEnd, ny) # Steering angles in degree
    DepthMm = image_data.depth
    imgDpth = np.linspace(0, DepthMm, nz) # Axial distance in mm
    volDepth = image_data.vol_depth # Coronal
    volWidth = image_data.vol_width # Lateral
    volHeight = image_data.vol_height # Axial
    fovSize   = [volWidth, volDepth, volHeight] # [Lateral, Elevation, Axial]
    imgSize = np.array(np.round(np.array([volWidth, volDepth, volHeight])*image_data.pix_per_mm), dtype=np.uint32) # [Lateral, Coronal, Axial]

    NonLinThr=3.5e4; NonLinDiv=1.7e4
    LinThr=3e4; LinDiv=3e4
    
    # Generate image
    imgOut = []; imgOutCoords = []
    if numVolumes > 1:
        for k in range(numVolumes):
            rxAngsAzVec = np.linspace(rxAngAz[0],rxAngAz[-1],dbEnvDatFullVolSeries[k].shape[1])
            rxAngsElVec = np.einsum('ikj->ijk', np.linspace(rxAngEl[0],rxAngEl[-1],dbEnvDatFullVolSeries[k].shape[2]))
            curImgOut, curImgOutCoords = scanConvert3Va(dbEnvDatFullVolSeries, rxAngsAzVec, rxAngsElVec, imgDpth,imgSize,fovSize, apexDist)
            if scale:
                if not isLin:
                    imgOut.append((curImgOut-NonLinThr)*255/NonLinDiv)
                else:
                    imgOut.append((curImgOut-LinThr)*255/LinDiv)
            else:
                curImgOut = np.array(curImgOut)
                if normalize:
                    curImgOut /= np.amax(curImgOut)
                    curImgOut *= 255
                imgOut.append(curImgOut)
                imgOutCoords.append(curImgOutCoords)
                
                
        imgOut = np.array(imgOut)
    else:
        rxAngsAzVec = np.linspace(rxAngAz[0],rxAngAz[-1],dbEnvDatFullVolSeries.shape[1])
        rxAngsElVec = np.linspace(rxAngEl[0],rxAngEl[-1],dbEnvDatFullVolSeries.shape[2])
        curImgOut, curImgOutCoords = scanConvert3Va(dbEnvDatFullVolSeries, rxAngsAzVec, rxAngsElVec, imgDpth,imgSize,fovSize, apexDist)
        if scale:
            if not isLin:
                imgOut = (curImgOut-NonLinThr)*255/NonLinDiv
            else:
                imgOut = (curImgOut-LinThr)*255/LinDiv
        else:
            curImgOut = np.array(curImgOut)
            if normalize:
                curImgOut /= np.amax(curImgOut)
                curImgOut *= 255
            imgOut = curImgOut
            imgOutCoords = curImgOutCoords
    
    return imgOut, imgOutCoords, fovSize