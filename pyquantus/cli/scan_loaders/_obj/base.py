import pickle

from abc import ABC

class ImageData(ABC):
    """
    Abstract base class for image data.
    """

    def __init__(self):
        pass

    def load(self, path: str):
        """
        Load RF image data from a given path.
        """
        if path.endswith('.pkl'):
            with open(path, 'rb') as file:
                self = pickle.load(file)
        else:
            raise ValueError("Unsupported file format. Only .pkl files are supported.")

    def save(self, path: str):
        """
        Save RF image data to a given path.
        """
        assert path.endswith('.pkl'), "Only .pkl files are supported."
        with open(path, 'wb') as file:
            pickle.dump(self, file)