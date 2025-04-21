import numpy as np
from dataclasses import dataclass

@dataclass
class OutImStruct():
    """Output image structure for scan converted images."""
    scArr: np.ndarray
    xmap: np.ndarray # sc (y,x) --> preSC x
    ymap: np.ndarray # sc (y,x) --> preSC y