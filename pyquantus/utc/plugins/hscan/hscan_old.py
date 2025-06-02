# Standard library imports
import logging
import os

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial
from scipy.fft import fft

# Local application imports
from src.data import Data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

######################################################################################

class Hscan:

    ###################################################################################
    # Constructor method (initializer)
    def __init__(self,
                 folder_path: str,  
                 save_path,      
                 device,
                 size,
                 signal_type,
                 ac_method: str, 
                 mode,
                 alpha
                 ) -> None:  

        # input arguments
        self.folder_path = folder_path
        self.save_path   = save_path
        self.device      = device
        self.size        = size
        self.signal_type = signal_type
        self.ac_method   = ac_method
        self.mode        = mode
        self.alpha       = alpha
         
        # initialize
        if   self.mode == "single":   self.__run_single()
        elif self.mode == "multiple": self.__run_multiple()

    ###################################################################################

    def __run_single(self):
                                    
        data_obj = Data(
                        sample_folder_path = self.folder_path,
                        device             = self.device,      # Device type: "L15" or "C3"  
                        size               = self.size,        # Size of the data: "large" or "small"
                        signal_type        = self.signal_type, # Signal type: "tgc" or "no_tgc"   
                        ac_method          = self.ac_method,   # Attenuation correction method options: "off", "fd_base_0.5", "afd_base","afda_base"
                        mode               = "read_data",      # Mode of operation: "read_data", "extract_data_s", "extract_data_m",  "roi_detection", "broadband"
                        visualize          = False,            # Visualization option: "on" or "off"
                        alpha              = self.alpha
                        )
        
        hscan_single_obj = HscanSingle(data_obj, visualize=False)
        self.save_hscan_single(hscan_single_obj, save_format="csv")
        
    ###################################################################################

    def __run_multiple(self):
        # Get the folders path inside self.folder_path
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]
        
        for folder_path in folder_paths:
            self.folder_path = folder_path
            try:
                self.__run_single()
            except Exception as e:
                logging.error(f"An error occurred while processing {folder_path}: {e}")
                # You can also log the stack trace if needed
                logging.exception("Exception occurred")

    ###################################################################################

    def save_hscan_single(self, hscan_single_obj, save_format):
        """
        Save the hscan_single object to either Excel or CSV format.

        Parameters:
        - hscan_single_obj: The object containing the data to be saved.
        - save_format: The format to save the data ('excel' or 'csv').
        """
        logging.info("Starting to save hscan_single object.")
        
        # Get the sample name from the folder path
        sample_name = os.path.basename(os.path.normpath(hscan_single_obj.data_obj.sample_folder_path))
        logging.info(f"Sample name extracted: {sample_name}")
        
        # Define the results directory path
        results_dir = os.path.join(self.save_path, 'results', sample_name)
        logging.info(f"Results directory set to: {results_dir}")

        # Create the results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        logging.info("Results directory created or already exists.")

        # Function to save frames for a given attribute in the specified format
        def save_attribute_frames(attribute_data, attribute_name):
            logging.info(f"Saving frames for attribute: {attribute_name}")
            
            if save_format == 'excel':
                excel_file_path = os.path.join(results_dir, f'{attribute_name}.xlsx')
                with pd.ExcelWriter(excel_file_path) as writer:
                    for frame_index in range(attribute_data.shape[2]):
                        # Create a DataFrame for the current frame
                        df = pd.DataFrame(attribute_data[:, :, frame_index])
                        # Add a column for the frame index
                        df['Frame_Index'] = frame_index
                        # Write the DataFrame to a new sheet
                        df.to_excel(writer, sheet_name=f'Frame_{frame_index}', index=False)
                        logging.info(f"Frame {frame_index} saved for attribute: {attribute_name} in Excel format.")

            elif save_format == 'csv':
                frames_list = []
                for frame_index in range(attribute_data.shape[2]):
                    # Create a DataFrame for the current frame
                    df = pd.DataFrame(attribute_data[:, :, frame_index])
                    # Add a column for the frame index
                    df['Frame_Index'] = frame_index
                    frames_list.append(df)

                # Concatenate all frames into a single DataFrame and save to a CSV file
                combined_df = pd.concat(frames_list, axis=0)
                csv_file_path = os.path.join(results_dir, f'{attribute_name}.csv')
                combined_df.to_csv(csv_file_path, index=False)
                logging.info(f"All frames saved for attribute: {attribute_name} in a single CSV file.")

        # Save attributes based on the specified format
        save_attribute_frames(hscan_single_obj.convolved_signal_with_ghx_1_3d, 'convolved_signal_with_ghx_1_3d')
        save_attribute_frames(hscan_single_obj.convolved_signal_with_ghx_2_3d, 'convolved_signal_with_ghx_2_3d')
        save_attribute_frames(hscan_single_obj.convolved_signal_with_ghx_1_envelope_3d, 'convolved_signal_with_ghx_1_envelope_3d')
        save_attribute_frames(hscan_single_obj.convolved_signal_with_ghx_2_envelope_3d, 'convolved_signal_with_ghx_2_envelope_3d')
        save_attribute_frames(hscan_single_obj.data_obj.trimmed_signal_3d, 'trimmed_signal_3d')
        save_attribute_frames(hscan_single_obj.data_obj.trimmed_signal_envelope_3d, 'trimmed_signal_envelope_3d')

        logging.info("Finished saving hscan_single object.")

    ###################################################################################


                                       
    
   


class HscanSingle:
    
    ######################################################################################
    # Constructor method (initializer)
    def __init__(self, data_obj, visualize) -> None:
        
        # necessary data after hscan
        self.data_obj:  object = data_obj
        self.visualize = visualize
        
        self.wavelet_1: object = None
        self.wavelet_2: object = None      
        self.convolved_signal_with_ghx_1_3d = None
        self.convolved_signal_with_ghx_2_3d = None
        self.convolved_signal_with_ghx_1_envelope_3d = None
        self.convolved_signal_with_ghx_2_envelope_3d = None
        self.convolved_signal_with_ghx_1_2d = None
        self.convolved_signal_with_ghx_2_2d = None
        self.convolved_signal_with_ghx_1_envelope_2d = None
        self.convolved_signal_with_ghx_2_envelope_2d = None
        self.convolved_signal_with_ghx_1_1d = None
        self.convolved_signal_with_ghx_2_1d = None
        self.convolved_signal_with_ghx_1_envelope_1d = None
        self.convolved_signal_with_ghx_2_envelope_1d = None

        self.__run()
       
    # main signal
    ###################################################################################

    def __run(self):
        
        # build wavelet
        self.build_wavelet()
        
        # convolved signals
        self.convolve_signal_with_wavelets_3d()
        self.get_envelope_of_convolved_signals_3d()
        
        self.convolve_signal_with_wavelets_2d()
        self.get_envelope_of_convolved_signals_2d()
        
        self.convolve_signal_with_wavelets_1d()
        self.get_envelope_of_convolved_signals_1d() 
        
    ###################################################################################

    def build_wavelet(self):
        logging.info("Starting wavelet build process.")

        # Create the first wavelet
        self.wavelet_1 = GaussinaHermiteWavelet(order=2,
                                 fs=self.data_obj.sampling_rate_MHz * 1e6,
                                 sigma=0.11e-6,
                                 wavelet_duration=3e-6,
                                 visualize=self.visualize)
        logging.info("Wavelet 1 created with parameters: order=2, fs=%s MHz, sigma=0.15 µs, wavelet_duration=3 µs, visualize=True", self.data_obj.sampling_rate_MHz)

        # Create the second wavelet
        self.wavelet_2 = GaussinaHermiteWavelet(order=8,
                                 fs=self.data_obj.sampling_rate_MHz * 1e6,
                                 sigma=0.11e-6,
                                 wavelet_duration=3e-6,
                                 visualize=self.visualize)
        logging.info("Wavelet 2 created with parameters: order=8, fs=%s MHz, sigma=0.15 µs, wavelet_duration=3 µs, visualize=True", self.data_obj.sampling_rate_MHz)

        logging.info("Wavelet build process completed.")
        
    ###################################################################################

    def convolve_3d_signal(self,
                           signal,
                           wavelet,
                           axis):
        """
        Convolve a 3D signal with a 1D wavelet along the specified axis.

        Parameters:
        - signal: np.ndarray
            A 3D array representing the signal to be convolved.
        - wavelet: np.ndarray
            A 1D array representing the wavelet to convolve with.
        - axis: int
            The axis along which to convolve.

        Returns:
        - np.ndarray
            The convolved 3D signal.
        """
        # Initialize an empty array for the convolved signal with the same shape as the original signal
        convolved_signal = np.zeros_like(signal)

        logging.info("Starting convolution along axis %d", axis)

        # Use np.apply_along_axis to perform convolution along the specified axis
        if axis == 0:
            convolved_signal = np.apply_along_axis(lambda x: np.convolve(x, wavelet, mode='same'), axis=0, arr=signal)
        elif axis == 1:
            convolved_signal = np.apply_along_axis(lambda x: np.convolve(x, wavelet, mode='same'), axis=1, arr=signal)
        elif axis == 2:
            convolved_signal = np.apply_along_axis(lambda x: np.convolve(x, wavelet, mode='same'), axis=2, arr=signal)
        else:
            logging.error("Invalid axis: %d. Axis must be 0, 1, or 2.", axis)
            raise ValueError("Axis must be 0, 1, or 2.")

        logging.info("Convolution completed.")
        return convolved_signal

    ###################################################################################
    
    def convolve_signal_with_wavelets_3d(self):
        logging.info("Starting convolution with wavelets")

        # Convolve with the first wavelet
        self.convolved_signal_with_ghx_1_3d = self.convolve_3d_signal(
            signal=self.data_obj.trimmed_signal_3d,
            wavelet=self.wavelet_1.wavelet,
            axis=1
        )
        logging.info("Convolution with wavelet 1 completed")
        
        # Convolve with the second wavelet
        self.convolved_signal_with_ghx_2_3d = self.convolve_3d_signal(
            signal=self.data_obj.trimmed_signal_3d,
            wavelet=self.wavelet_2.wavelet,
            axis=1
        )
        logging.info("Convolution with wavelet 2 completed")
                
    ###################################################################################

    def get_envelope_of_convolved_signals_3d(self):
        logging.info("Starting to compute the envelope of convolved signals in 3D.")

        try:
            logging.info("Calculating envelope for the first convolved signal.")
            self.convolved_signal_with_ghx_1_envelope_3d = Data.get_signal_envelope_xd(
                self.convolved_signal_with_ghx_1_3d,  # Input signal
                hilbert_transform_axis=self.data_obj.hilbert_transform_axis # Axis along which to apply the Hilbert transform
            )
            logging.info("Envelope for the first convolved signal computed successfully.")

            logging.info("Calculating envelope for the second convolved signal.")
            self.convolved_signal_with_ghx_2_envelope_3d = Data.get_signal_envelope_xd(
                self.convolved_signal_with_ghx_2_3d,  # Input signal
                hilbert_transform_axis=self.data_obj.hilbert_transform_axis # Axis along which to apply the Hilbert transform
            )
            logging.info("Envelope for the second convolved signal computed successfully.")

        except Exception as e:
            logging.error(f"An error occurred while computing the envelopes: {e}")

    ###################################################################################

    def convolve_signal_with_wavelets_2d(self):
        logging.debug("Starting 2D convolution with wavelets.")
        
        try:
            self.convolved_signal_with_ghx_1_2d = self.convolved_signal_with_ghx_1_3d[:, :, self.data_obj.default_frame]
            logging.debug("Convolution with ghx_1 completed. Shape: %s", self.convolved_signal_with_ghx_1_2d.shape)
            
            self.convolved_signal_with_ghx_2_2d = self.convolved_signal_with_ghx_2_3d[:, :, self.data_obj.default_frame]
            logging.debug("Convolution with ghx_2 completed. Shape: %s", self.convolved_signal_with_ghx_2_2d.shape)
        
        except Exception as e:
            logging.error("Error during convolution: %s", e)
            raise  # Re-raise the exception after logging it
        finally:
            logging.debug("2D convolution process completed.")
        
    ###################################################################################

    def get_envelope_of_convolved_signals_2d(self):
        logging.debug("Starting to get the envelope of convolved signals in 2D.")
        
        try:
            self.convolved_signal_with_ghx_1_envelope_2d = self.convolved_signal_with_ghx_1_envelope_3d[:, :, self.data_obj.default_frame]
            logging.debug("Envelope extraction for ghx_1 completed. Shape: %s", self.convolved_signal_with_ghx_1_envelope_2d.shape)
            
            self.convolved_signal_with_ghx_2_envelope_2d = self.convolved_signal_with_ghx_2_envelope_3d[:, :, self.data_obj.default_frame]
            logging.debug("Envelope extraction for ghx_2 completed. Shape: %s", self.convolved_signal_with_ghx_2_envelope_2d.shape)
        
        except Exception as e:
            logging.error("Error during envelope extraction: %s", e)
            raise  # Re-raise the exception after logging it
        finally:
            logging.debug("Envelope extraction process completed.")

    ###################################################################################

    def convolve_signal_with_wavelets_1d(self):
        logging.debug("Starting 1D convolution with wavelets.")
        
        try:
            # Convolve wavelet with trimmed signal using GH2 and GH8 wavelets
            self.convolved_signal_with_ghx_1_1d = self.convolved_signal_with_ghx_1_2d[self.data_obj.ROI_analysis_vline_0d, :]
            logging.debug("Convolution with ghx_1 completed. Shape: %s", self.convolved_signal_with_ghx_1_1d.shape)

            self.convolved_signal_with_ghx_2_1d = self.convolved_signal_with_ghx_2_2d[self.data_obj.ROI_analysis_vline_0d, :]
            logging.debug("Convolution with ghx_2 completed. Shape: %s", self.convolved_signal_with_ghx_2_1d.shape)

        except Exception as e:
            logging.error("Error during 1D convolution: %s", e)
            raise  # Re-raise the exception after logging it
        finally:
            logging.debug("1D convolution process completed.")

    ###################################################################################

    def get_envelope_of_convolved_signals_1d(self):
        logging.debug("Starting to get the envelope of convolved signals in 1D.")
        
        try:
            self.convolved_signal_with_ghx_1_envelope_1d = self.convolved_signal_with_ghx_1_envelope_2d[self.data_obj.ROI_analysis_vline_0d, :]
            logging.debug("Envelope extraction for ghx_1 completed. Shape: %s", self.convolved_signal_with_ghx_1_envelope_1d.shape)

            self.convolved_signal_with_ghx_2_envelope_1d = self.convolved_signal_with_ghx_2_envelope_2d[self.data_obj.ROI_analysis_vline_0d, :]
            logging.debug("Envelope extraction for ghx_2 completed. Shape: %s", self.convolved_signal_with_ghx_2_envelope_1d.shape)

        except Exception as e:
            logging.error("Error during envelope extraction in 1D: %s", e)
            raise  # Re-raise the exception after logging it
        finally:
            logging.debug("Envelope extraction process in 1D completed.")

    ###################################################################################


   

class GaussinaHermiteWavelet:
    
    ###################################################################################
    # Constructor method (initializer)
    def __init__(self, order, fs, sigma, wavelet_duration, visualize) -> None:
                
        self.order = order
        self.fs = fs 
        self.sigma = sigma
        self.wavelet_duration = wavelet_duration
        self.visualize = visualize
    
        self.time = None
        self.wavelet = None

        self.__run()
        
    ###################################################################################

    def __run(self):
        
        self.build_time_array()
        self.build_wavelet()
        self.get_central_freq_of_wavelet()
        self.plot()
    
    ###################################################################################
    
    def build_time_array(self):
        logging.info("Building time array...")

        # Get the sampling frequency rate
        time_step_second = 1 / (self.fs)
        logging.debug(f"Calculated time step (seconds): {time_step_second}")

        # Create a time array for the wavelet
        # Ensure the wavelet duration is a float for np.arange
        half_duration = self.wavelet_duration / 2  # Use float division
        self.time = np.arange(-half_duration, half_duration, time_step_second)
        
        logging.info(f"Time array created with length: {len(self.time)}")
        
    ###################################################################################
    
    def build_wavelet(self):
        logging.debug(f"Starting normalized_hermite_polynomial with order={self.order}, time={self.time}, sigma={self.sigma}")

        # Physicists' Hermite polynomial
        H_n = hermite(self.order)
        logging.debug(f"Hermite polynomial of order {self.order} generated.")

        # Normalization factor (assuming custom energy calculation)
        energy = np.sqrt(np.pi / 2) * (factorial(2 * self.order) / (2**self.order * factorial(self.order)))
        normalization_factor = 1.0 / np.sqrt(energy)
        logging.debug(f"Calculated energy={energy} and normalization_factor={normalization_factor}")

        # Gaussian window
        gaussian_window = np.exp(-(self.time**2) / (1 * self.sigma**2))
        logging.debug(f"Computed Gaussian window with values: {gaussian_window}")

        # Construct the wavelet
        self.wavelet = normalization_factor * H_n(self.time / self.sigma) * gaussian_window
        logging.debug(f"Generated wavelet with values: {self.wavelet}")

    ###################################################################################

    def get_central_freq_of_wavelet(self):
        logging.info("Starting central frequency calculation.")
        
        # Perform Fourier Transform
        fft_wavelet = np.fft.fft(self.wavelet)
        logging.debug(f"FFT of wavelet: {fft_wavelet}")

        # Convert sampling rate to Hz and calculate dt
        dt = 1 / self.fs

        # Compute the frequency bins
        n = len(self.wavelet)
        freq_bins = np.fft.fftfreq(n, d=dt)
        logging.debug(f"Frequency bins: {freq_bins}")

        # Compute the Power Spectrum
        power_spectrum = np.abs(fft_wavelet) ** 2
        logging.debug(f"Power Spectrum: {power_spectrum}")

        # Limit to positive frequencies
        positive_freqs = freq_bins > 0
        positive_freq_bins = freq_bins[positive_freqs]
        positive_power_spectrum = power_spectrum[positive_freqs]

        # Find the frequency with maximum amplitude in the positive spectrum
        max_amplitude_freq = positive_freq_bins[np.argmax(positive_power_spectrum)]
        logging.info(f"Frequency with maximum amplitude: {max_amplitude_freq / 1e6:.2f} MHz")

        # Replace print with logging
        logging.info(f"Frequency with Maximum Amplitude: {max_amplitude_freq / 1e6:.2f} MHz")

    ###################################################################################

    def plot(self, show_negative_freqs=False):
        logging.info("Starting the plot process.")
        
        if self.visualize:
            logging.info("Visualization is enabled.")
            
            plt.figure(figsize=(12, 4))
            logging.info("Figure created with size 12x4.")

            # Plot the Hermite wavelet
            plt.subplot(1, 2, 1)
            plt.plot(self.time * 1e6, self.wavelet, label=f'GH{self.order}')  # Time in µs
            plt.axvline(x=3 * self.sigma * 1e6, color='blue', linestyle='--', label='+3σ')
            plt.axvline(x=-3 * self.sigma * 1e6, color='blue', linestyle='--', label='-3σ')
            plt.title('Hermite Wavelets')
            plt.xlabel('Time [µs]')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            logging.info("Hermite wavelet plot created with time scaled to microseconds.")

            # FFT and frequency response
            logging.info("Starting FFT calculation.")
            wavelet_fft = fft(self.wavelet)
            freqs = np.fft.fftfreq(len(wavelet_fft), d=(self.time[1] - self.time[0]))  # Frequency vector
            fft_magnitude = np.abs(wavelet_fft)  # Magnitude of FFT
            logging.info("FFT calculation completed.")

            plt.subplot(1, 2, 2)
            if show_negative_freqs:
                plt.plot(freqs / 1e6, fft_magnitude, label=f'FFT of GH{self.order}')  # Full spectrum
                plt.title('Full Frequency Spectrum of Hermite Wavelets')
            else:
                positive_mask = freqs >= 0
                plt.plot(freqs[positive_mask] / 1e6, fft_magnitude[positive_mask], label=f'FFT of GH{self.order}')  # Positive only
                plt.title('Positive Frequency Spectrum of Hermite Wavelets')
            
            plt.xlabel('Frequency [MHz]')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.grid()
            logging.info("Frequency response plot created.")

            plt.tight_layout()
            plt.show()
            logging.info("Plots displayed successfully.")
        else:
            logging.info("Visualization is disabled; no plots will be shown.")

    ###################################################################################
    
    
    
    

