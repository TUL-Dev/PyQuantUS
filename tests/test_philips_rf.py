import os
import unittest
import numpy as np
from pathlib import Path
from pyquantus.parse.philipsRf import PhilipsRFParser

class TestPhilipsRFParser(unittest.TestCase):
    def setUp(self):
        self.parser = PhilipsRFParser()
        self.test_file = None  # Will be set if test file exists
        
        # Check if test file exists in the test data directory
        test_data_dir = Path(__file__).parent / 'test_data'
        if test_data_dir.exists():
            for file in test_data_dir.glob('*.rf'):
                self.test_file = str(file)
                break

    def test_initialization(self):
        """Test that the parser initializes correctly"""
        self.assertIsNotNone(self.parser)
        self.assertEqual(len(self.parser.VHeader), 20)
        self.assertEqual(len(self.parser.FHeader), 20)
        self.assertEqual(self.parser.fileHeaderSize, 20)

    def test_find_signature(self):
        """Test signature detection"""
        if self.test_file:
            sig = self.parser.find_signature(Path(self.test_file))
            self.assertEqual(len(sig), 8)
            self.assertEqual(sig, [0, 0, 0, 0, 255, 255, 0, 0])
        else:
            self.skipTest("No test file available")

    def test_parse_file_header(self):
        """Test file header parsing"""
        if self.test_file:
            with open(self.test_file, 'rb') as f:
                dbParams, numFileHeaderBytes = self.parser.parse_file_header(f, 'big')
                self.assertIsNotNone(dbParams)
                self.assertIsInstance(numFileHeaderBytes, int)
                self.assertGreater(numFileHeaderBytes, 0)
        else:
            self.skipTest("No test file available")

    def test_parse_rf(self):
        """Test RF data parsing"""
        if self.test_file:
            rfdata = self.parser.parse_rf(self.test_file, 0, 2000)
            self.assertIsNotNone(rfdata)
            self.assertIsNotNone(rfdata.lineData)
            self.assertIsNotNone(rfdata.lineHeader)
            self.assertIsNotNone(rfdata.headerInfo)
            self.assertIsNotNone(rfdata.dbParams)
        else:
            self.skipTest("No test file available")

    def test_parse(self):
        """Test the main parse method"""
        if self.test_file:
            shape = self.parser.parse(self.test_file)
            self.assertIsInstance(shape, tuple)
            self.assertEqual(len(shape), 4)  # Should return (numFrame, NumSonoCTAngles, pt, ML_out * txBeamperFrame)
            
            # Check if .mat file was created
            mat_file = self.test_file.rsplit('.', 1)[0] + '.mat'
            self.assertTrue(os.path.exists(mat_file))
        else:
            self.skipTest("No test file available")

if __name__ == '__main__':
    unittest.main() 