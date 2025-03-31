from pathlib import Path
from typing import Tuple

from pyquantus.utc import UltrasoundImage
from pyquantus.parse.objects import ScConfig
from pyquantus.parse.clarius import ClariusTarUnpacker, clariusRfParser
from pyquantus.parse.canon import canonIqParser
from pyquantus.parse.siemens import siemensRfParser
from pyquantus.parse.terason import terasonRfParser

def parse_canon_iq(scan_path: str, phantom_path: str, frame: int) -> UltrasoundImage:
    # Load signal data
    imgData, imgInfo, refData, refInfo = canonIqParser(scan_path, phantom_path)
    
    # Package data
    ultrasoundImage = UltrasoundImage()
    ultrasoundImage.axialResRf = imgInfo.depth / imgData.rf.shape[0]
    ultrasoundImage.lateralResRf = ultrasoundImage.axialResRf * (
        imgData.rf.shape[0]/imgData.rf.shape[1]
    ) # placeholder
    ultrasoundImage.bmode = imgData.bMode
    ultrasoundImage.phantomRf = refData.rf
    ultrasoundImage.rf = imgData.rf
    ultrasoundImage.scBmode = imgData.scBmode
    ultrasoundImage.xmap = imgData.scBmodeStruct.xmap
    ultrasoundImage.ymap = imgData.scBmodeStruct.ymap
    scConfig = ScConfig(imgInfo.width1, imgInfo.tilt1, imgInfo.startDepth1, imgInfo.endDepth1, imgInfo.numSamplesDrOut)
    ultrasoundImage.scConfig = scConfig
    
    return ultrasoundImage
    

def parse_siemens_rf(scan_path: str, phantom_path: str, frame: int) -> UltrasoundImage:
    # Load signal data
    imgData, imgInfo, refData, refInfo = siemensRfParser(scan_path, phantom_path)
    
    # Package data
    ultrasoundImage = UltrasoundImage()
    ultrasoundImage.axialResRf = imgInfo.depth / imgData.rf[frame].shape[0]
    ultrasoundImage.lateralResRf = ultrasoundImage.axialResRf * (
        imgData.rf[frame].shape[0]/imgData.rf[frame].shape[1]
    ) # placeholder
    ultrasoundImage.bmode = imgData.bMode[frame]
    ultrasoundImage.phantomRf = refData.rf[0]
    ultrasoundImage.rf = imgData.rf[frame]
    
    return ultrasoundImage

def parse_terason_rf(scan_path: str, phantom_path: str, frame: int) -> UltrasoundImage:
    imgData, imgInfo, refData, refInfo = terasonRfParser(scan_path, phantom_path)
    
    # Package data
    ultrasoundImage = UltrasoundImage()
    ultrasoundImage.axialResRf = imgInfo.axialRes
    ultrasoundImage.lateralResRf = imgInfo.lateralRes
    ultrasoundImage.bmode = imgData.bMode
    ultrasoundImage.phantomRf = refData.rf
    ultrasoundImage.rf = imgData.rf
    
    return ultrasoundImage

def parse_clarius_rf(imgFilename: str, phantomFilename: str, frame: int) -> UltrasoundImage:
    """Parse Clarius RF data and metadata from inputted files. Assumes consistent naming
    conventions across RAW, TGC, and YML files.

    Args:
        imgFilename (str): File path of the RF data (.raw).
        phantomFilename (str): File path of the phantom RF data.

    Returns:
        Tuple: Image data, image metadata, phantom data, and phantom metadata.
    """
    if imgFilename.endswith(".tar"):
        ClariusTarUnpacker(imgFilename, "single_tar")
        unpackedTarFolder = Path(imgFilename.replace(".tar", "_extracted"))
        imageRfPath = ""; imageTgcPath = ""; imageInfoPath = ""
        for file in unpackedTarFolder.iterdir():
            if file.name.endswith("_rf.raw"):
                imageRfPath = str(file)
            elif file.name.endswith("_env.tgc.yml"):
                imageTgcPath = str(file)
            elif file.name.endswith("_rf.yml"):
                imageInfoPath = str(file)
        if imageRfPath == "" or imageInfoPath == "":
            raise Exception("Missing files in tar")
    else:
        imageRfPath = imgFilename
        imageInfoPath = imageRfPath.replace(".raw", ".yml")
        imageTgcPath = imageRfPath.replace("_rf.raw", "_env.tgc.yml")
    if not len(imageTgcPath) or not Path(imageTgcPath).exists():
        imageTgcPath = None
    if phantomFilename.endswith(".tar"):
        ClariusTarUnpacker(phantomFilename, "single_tar")
        unpackedTarFolder = Path(phantomFilename.replace(".tar", "_extracted"))
        phantomRfPath = ""; phantomTgcPath = ""; phantomInfoPath = ""
        for file in unpackedTarFolder.iterdir():
            if file.name.endswith("_rf.raw"):
                phantomRfPath = str(file)
            elif file.name.endswith("_env.tgc.yml"):
                phantomTgcPath = str(file)
            elif file.name.endswith("_rf.yml"):
                phantomInfoPath = str(file)
        if phantomRfPath == "" or phantomInfoPath == "":
            raise Exception("Missing files in tar")
    else:
        phantomRfPath = phantomFilename
        phantomInfoPath = phantomRfPath.replace(".raw", ".yml")
        phantomTgcPath = phantomRfPath.replace("_rf.raw", "_env.tgc.yml")
    if not len(phantomTgcPath) or not Path(phantomTgcPath).exists():
        phantomTgcPath = None
        
    imgData, imgInfo, refData, refInfo, sc = clariusRfParser(imageRfPath, imageTgcPath, imageInfoPath, phantomRfPath, phantomTgcPath, phantomInfoPath)
    
    # Package data
    ultrasoundImage = UltrasoundImage()
    ultrasoundImage.axialResRf = imgInfo.depth / imgData.rf[frame].shape[0]
    ultrasoundImage.lateralResRf = ultrasoundImage.axialResRf * (
        imgData.rf[frame].shape[0]/imgData.rf[frame].shape[1]
    ) # placeholder
    ultrasoundImage.bmode = imgData.bMode[frame]
    ultrasoundImage.scBmode = imgData.scBmode[frame]
    ultrasoundImage.phantomRf = refData.rf[0]
    ultrasoundImage.rf = imgData.rf[frame]
    ultrasoundImage.xmap = imgData.scBmodeStruct.xmap
    ultrasoundImage.ymap = imgData.scBmodeStruct.ymap
    
    if sc:
        scConfig = ScConfig()
        scConfig.width = imgInfo.width1
        scConfig.tilt = imgInfo.tilt1
        scConfig.startDepth = imgInfo.startDepth1
        scConfig.endDepth = imgInfo.endDepth1
        ultrasoundImage.scConfig = scConfig
    
    return ultrasoundImage
