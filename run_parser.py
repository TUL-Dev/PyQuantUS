from pyquantus.parse.philipsRf import PhilipsRFParser
import os

def main():
    filepath = r"d:\Omid\0_samples\Philips\UKDFIBEPIC003\UKDFIBEPIC003INTER3D_20250424_094008.rf"
    
    # Check file signature
    with open(filepath, 'rb') as f:
        sig = list(f.read(8))
        print(f"File signature: {sig}")
        assert sig == [0, 0, 0, 0, 255, 255, 0, 0], "Invalid file signature"
    
    # Parse RF file
    parser = PhilipsRFParser()
    result = parser.parse(filepath)
    print(f"Processing completed. Output shape: {result}")
    
    # Check if .mat file was created
    mat_file = filepath.rsplit('.', 1)[0] + '.mat'
    if os.path.exists(mat_file):
        print(f"Created .mat file: {mat_file}")

if __name__ == "__main__":
    main() 