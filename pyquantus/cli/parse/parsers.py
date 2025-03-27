from pathlib import Path
from typing import Tuple

from pyquantus.parse.canon import canonIqParser
from pyquantus.parse.clarius import ClariusTarUnpacker, clariusRfParser
from pyquantus.parse.philipsMat import philips2dRfMatParser
from pyquantus.parse.philipsRf import philipsRfParser
from pyquantus.parse.siemens import siemensRfParser
from pyquantus.parse.terason import terasonRfParser
from pyquantus.parse.verasonics import verasonicsRfParser

def parseCanonIq(scan_path: str, phantom_path: str) -> None:
    return canonIqParser(scan_path, phantom_path)

def parseSiemensRf(scan_path: str, phantom_path: str) -> None:
    return siemensRfParser(scan_path, phantom_path)

def parseTerasonRf(scan_path: str, phantom_path: str) -> None:
    return terasonRfParser(scan_path, phantom_path)

def parseClariusRf(imgFilename: str, phantomFilename: str):
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
        
    return clariusRfParser(imageRfPath, imageTgcPath, imageInfoPath, phantomRfPath, phantomTgcPath, phantomInfoPath)