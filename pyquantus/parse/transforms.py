import numpy as np
from numpy.matlib import repmat

from pyquantus.parse.objects import OutImStruct

def iqToRf(iqData, rxFrequency, decimationFactor, carrierFrequency):
    import scipy.signal as ssg    
    iqData = ssg.resample_poly(iqData, decimationFactor, 1) # up-sample by decimation factor
    rfData = np.zeros(iqData.shape)
    t = [i*(1/rxFrequency) for i in range(iqData.shape[0])]
    for i in range(iqData.shape[1]):
        rfData[:,i] = np.real(np.multiply(iqData[:,i], np.exp(1j*(2*np.pi*carrierFrequency*np.transpose(t)))))
    return rfData

def scanConvert(inIm, width, tilt, startDepth, stopDepth, desiredHeight=500):
    # ScanConvert sector image
    # Inputs:
    #      InIm          Input image
    #      Width         Sector width of input image in degrees 
    #      Tilt          Tilt of sector image in degrees  
    #      StartDepth    Axial depth of first sample in meters 
    #      EndDepth      Axial depth of last sample in meters   
    #      DesiredHeight Desired vertical size of output image in pixels 
    #      (default 500)
    # 
    #    Outputs:
    #      OutIm         Output (scanconverted) image(s)  
    #      HCm,WCm       Height and Width of image in centimeters 

    #Convert to radians
    samples, beams = inIm.shape
    depthIncrement = (stopDepth-startDepth)/(samples - 1)
    startAngle = np.deg2rad(270 + tilt - width/2)
    angleIncrement = np.deg2rad(width)/(beams-1)

    outIm = inIm
    background = 0
    height = desiredHeight

    # Subtract 180 degrees to get transudcer in top of image if startAngle > pi
    startAngle = startAngle % np.pi
    stopAngle = startAngle + np.deg2rad(width)
    angleRange = np.arange(startAngle, stopAngle+angleIncrement, angleIncrement)

    # Define physical limits of image
    xmin = -1*max(np.cos(startAngle)*np.array([startDepth, stopDepth]))
    xmax = -1*min(np.cos(stopAngle)*np.array([startDepth, stopDepth]))
    ymin = min(np.sin(angleRange)*startDepth)
    ymax = max(np.sin(angleRange)*stopDepth)
    widthScale = abs((xmax-xmin)/(ymax-ymin))
    width = int(np.ceil(height*widthScale))

    heightCm = (ymax-ymin)*100
    widthCm = (xmax-xmin)*100

    # Make (x,y)-plane representation of physical image
    xmat = (np.transpose(np.ones((1, height)))*np.arange(0, width, 1))/(width-1)
    ymat = (np.transpose(np.arange(0, height, 1).reshape((1, height)))*np.ones((1, width)))/(height-1)
    xmat = (xmat*(xmax-xmin)) + xmin
    ymat = (ymat*(ymax-ymin)) + ymin

    # Transform into polar coordinates (angle, range)
    anglemat = np.arctan2(ymat, -1*xmat)
    rmat = np.sqrt((xmat**2) + (ymat**2))

    # Convert phys. angle and range into beam and sample
    anglemat = np.ceil((anglemat - startAngle)/angleIncrement)
    rmat = np.ceil((rmat - startDepth)/depthIncrement)

    # Find pixels outside active sector
    backgr = np.argwhere((rmat<1))
    backgr = np.concatenate((backgr, np.argwhere(rmat>=samples)), axis=0)
    backgr = np.concatenate((backgr, np.argwhere(anglemat<1)), axis=0)
    backgr = np.concatenate((backgr, np.argwhere(anglemat>beams)), axis=0)
    scMap = (anglemat-1)*samples + rmat

    for i in range(backgr.shape[0]):
        scMap[backgr[i,0],backgr[i,1]] = background
    inIm_indx = repmat(np.arange(0, outIm.shape[1]), int(outIm.shape[0]), 1) # <-- maps (y,x) in Iin to indt in Iin
    inIm_indy = np.transpose(repmat(np.arange(0, outIm.shape[0]), int(outIm.shape[1]), 1)) # <-- maps (y,x) in Iout to indr in Iin
        
    outIm = np.append(np.transpose(outIm), background)
    inIm_indy = np.append(np.transpose(inIm_indy), background)
    inIm_indx = np.append(np.transpose(inIm_indx), background)

    scMap = np.array(scMap).astype(np.uint64) - 1
    outIm = outIm[scMap]
    inIm_indy = inIm_indy[scMap]
    inIm_indx = inIm_indx[scMap]

    outIm = np.reshape(outIm, (height, width))
    inIm_indy = np.reshape(inIm_indy, (height, width))
    inIm_indx = np.reshape(inIm_indx, (height, width))
    
    hCm = heightCm
    wCm = widthCm
    
    OutIm = OutImStruct()
    OutIm.scArr = outIm
    OutIm.preScArr = inIm
    OutIm.ymap = inIm_indy
    OutIm.xmap = inIm_indx
    return OutIm, hCm, wCm