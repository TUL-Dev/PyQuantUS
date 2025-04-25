import math
import numpy as np
from typing import Tuple
from scipy.interpolate import interpn


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

def scanConvert3dVolumeSeries(dbEnvDatFullVolSeries, scParams, isLin=True, scale=True, interp='linear', normalize=True) -> Tuple[np.ndarray, np.ndarray, list]:
    if len(dbEnvDatFullVolSeries.shape) != 4:
        numVolumes = 1
        nz, nx, ny = dbEnvDatFullVolSeries.shape
    else:
        numVolumes = dbEnvDatFullVolSeries.shape[0]
        nz, nx, ny = dbEnvDatFullVolSeries[0].shape
    apexDist = scParams.VDB_2D_ECHO_APEX_TO_SKINLINE # Distance of virtual apex to probe surface in mm
    azimSteerAngleStart = scParams.VDB_2D_ECHO_START_WIDTH_GC*180/np.pi # Azimuth steering angle (start) in degree
    azimSteerAngleEnd = scParams.VDB_2D_ECHO_STOP_WIDTH_GC*180/np.pi # Azimuth steering angle (end) in degree
    rxAngAz = np.linspace(azimSteerAngleStart, azimSteerAngleEnd, nx) # Steering angles in degree
    elevSteerAngleStart = scParams.VDB_THREED_START_ELEVATION_ACTUAL*180/np.pi # Elevation steering angle (start) in degree
    elevSteerAngleEnd = scParams.VDB_THREED_STOP_ELEVATION_ACTUAL*180/np.pi # Elevation steering angle (end) in degree
    rxAngEl = np.linspace(elevSteerAngleStart, elevSteerAngleEnd, ny) # Steering angles in degree
    DepthMm=scParams.VDB_2D_ECHO_STOP_DEPTH_SIP
    imgDpth = np.linspace(0, DepthMm, nz) # Axial distance in mm
    volDepth = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP *(abs(math.sin(math.radians(elevSteerAngleStart))) + abs(math.sin(math.radians(elevSteerAngleEnd)))) # Elevation (needs validation)
    volWidth = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP *(abs(math.sin(math.radians(azimSteerAngleStart))) + abs(math.sin(math.radians(azimSteerAngleEnd))))   # Lateral (needs validation)
    volHeight = scParams.VDB_2D_ECHO_STOP_DEPTH_SIP - scParams.VDB_2D_ECHO_START_DEPTH_SIP # Axial (needs validation)
    fovSize   = [volWidth, volDepth, volHeight] # [Lateral, Elevation, Axial]
    imgSize = np.array(np.round(np.array([volWidth, volDepth, volHeight])*scParams.pixPerMm), dtype=np.uint32) # [Lateral, Elevation, Axial]

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