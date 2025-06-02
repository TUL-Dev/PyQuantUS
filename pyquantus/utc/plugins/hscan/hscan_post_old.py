# Standard library imports
import logging
import os

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Configure logging if not already configured
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

######################################################################################


class HscanPostSteatosis:
    
    def __init__(self, folder_path, mode, method) -> None:
    
        self.folder_path = folder_path
        self.mode = mode
        self.method = method
        
        self.CAP_E_csv_file_path = r'data\source_files\CAP_E.csv' 
        self.df = None

        # initialize
        if   self.mode == "single":   self.__run_single()
        elif self.mode == "multiple": self.__run_multiple()

    ###################################################################################

    def __run_single(self):
        # Set up logging configuration (this can also be done in the main part of your code)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        try:
            logging.info(f"Starting processing for single folder: {self.folder_path}")

            # Create an instance of HscanSinglePost with visualize=True
            hscan_post_obj = HscanSinglePost(folder_path=self.folder_path,
                                              mode=self.mode,
                                              method=self.method,
                                              visualize=True)

            logging.info(f"Successfully processed folder: {self.folder_path}")

        except Exception as e:
            logging.error(f"An error occurred while processing {self.folder_path}: {e}")
            logging.exception("Exception occurred")
            
    ###################################################################################

    def __run_multiple(self):
        # Set up logging configuration (this can also be done in the main part of your code)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Get the folders path inside self.folder_path
        folder_paths = [os.path.join(self.folder_path, folder) for folder in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, folder))]
        
        ration_BR_array = []
        folder_paths_array = []
        
        logging.info(f"Found {len(folder_paths)} folders in {self.folder_path}.")
         
        for folder_path in folder_paths:
            try:
                logging.info(f"Processing folder: {folder_path}")
                
                hscan_post_obj = HscanSinglePost(folder_path=folder_path,
                                                  mode=self.mode,
                                                  method=self.method,
                                                  visualize=False)  
                
                R_double_start = np.sum(hscan_post_obj.R_channel_2d)
                G_double_start = np.sum(hscan_post_obj.G_channel_2d)
                B_double_start = np.sum(hscan_post_obj.B_channel_2d)

                ration_BR = B_double_start / R_double_start
                
                ration_BR_array.append(ration_BR)
                folder_paths_array.append(folder_path)
                
                logging.info(f"Calculated Ratio_BR for {folder_path}: {ration_BR:.4f}")
                       
            except Exception as e:
                logging.error(f"An error occurred while processing {folder_path}: {e}")
                logging.exception("Exception occurred")

        # Create a DataFrame from the appended arrays
        data = {
            'Folder_Path': folder_paths_array,
            'Ratio_BR': ration_BR_array,
            'CAP': None,
            'E': None,
            'LMH': None
        }
        
        self.df = pd.DataFrame(data)
        logging.info("DataFrame created with calculated Ratio_BR values.")
        
        self.fill_CAP_E_LMH_to_dataframe()
        logging.info("Filled CAP, E, LMH into the DataFrame.")
        
        self.plot_scatter_with_tests()
        logging.info("Scatter plot generated with tests.")

    ###################################################################################

    def fill_CAP_E_LMH_to_dataframe(self,
                                    limit_1: int = 248,
                                    limit_2: int = 280) -> pd.DataFrame:
        
        logging.info("Starting to fill CAP, E, and LMH into the DataFrame.")

        # Read the CSV file into a DataFrame df_CAP_E
        try:
            df_CAP_E = pd.read_csv(self.CAP_E_csv_file_path, delimiter=';')  
            logging.info(f"CSV file {self.CAP_E_csv_file_path} loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to read CSV file: {e}")
            return self.df  # Return the original DataFrame on failure

        # Iterate through rows of df_CAP_E
        for index_1 in df_CAP_E.index:  
            patient_id = df_CAP_E.Patient_ID[index_1]  
            logging.debug(f"Processing Patient ID: {patient_id}")

            # Iterate through rows of self.df
            for index_2 in self.df.index:  
                # Check if patient ID from df_CAP_E exists in self.df's 'Folder_Path' column
                if patient_id in self.df.Folder_Path[index_2]:  
                    logging.debug(f"Found matching patient ID {patient_id} in self.df at index {index_2}.")

                    # Assign Median_CAP and Median_E values from df_CAP_E to corresponding rows in self.df
                    self.df.loc[index_2, 'CAP'] = df_CAP_E['Median_CAP'][index_1]
                    self.df.loc[index_2, 'E'] = df_CAP_E['Median_E'][index_1]

                    # Convert DataFrame type to string
                    self.df['LMH'] = self.df['LMH'].astype(str)
                    
                    # Assign LMH category based on CAP value
                    if 0 <= self.df.loc[index_2, 'CAP'] < limit_1:
                        self.df.loc[index_2, 'LMH'] = "Low"
                    elif limit_1 <= self.df.loc[index_2, 'CAP'] < limit_2:
                        self.df.loc[index_2, 'LMH'] = "Medium"
                    elif limit_2 <= self.df.loc[index_2, 'CAP']:
                        self.df.loc[index_2, 'LMH'] = "High"

        logging.info("Completed filling CAP, E, and LMH into the DataFrame.")
        return self.df
        
    ###################################################################################

    def plot_scatter_with_tests(self,
                                limit_1: int = 248,
                                limit_2: int = 280,
                                dot_size: int = 50):  # Added a parameter for dot size

        logging.info("Starting scatter plot generation.")

        plt.rcdefaults()
        plt.figure(figsize=(10, 6))  # Set the figure size

        # Convert x and y to numeric data
        try:
            x = pd.to_numeric(self.df.CAP, errors='raise')
            y = pd.to_numeric(self.df.Ratio_BR, errors='raise')
            logging.info("Conversion of CAP and Ratio_BR to numeric succeeded.")
        except Exception as e:
            logging.error(f"Error converting CAP and Ratio_BR to numeric: {e}")
            return

        # Define conditions for different regions
        condition_1 = (0 <= x) & (x < limit_1)
        condition_2 = (limit_1 <= x) & (x < limit_2)
        condition_3 = (limit_2 <= x)

        logging.debug(f"Conditions defined: limit_1={limit_1}, limit_2={limit_2}")

        # Plot the scatter plot with different colors for different regions and increased dot size
        plt.scatter(x[condition_1], y[condition_1], marker='.', color='red', s=dot_size)
        plt.scatter(x[condition_2], y[condition_2], marker='.', color='black', s=dot_size)
        plt.scatter(x[condition_3], y[condition_3], marker='.', color='blue', s=dot_size)

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        logging.info("Correlation coefficient calculated successfully.")

        # Plot the line of slope (optional)
        slope, intercept = np.polyfit(x, y, 1)
        plt.plot(x, slope * x + intercept, color='green', label='Line of Fit')

        # Set titles and labels with correlation coefficient in the title
        plt.title(f'Scatter Plot of CAP vs. Ratio_BR (Correlation: {correlation:.2f})')
        plt.xlabel('CAP')
        plt.ylabel('Ratio BR')
        plt.legend()
        plt.grid(True)  # Add grid for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping

        logging.info("Displaying the scatter plot.")
        plt.show()

    ###################################################################################


    
    
    
class HscanSinglePost:

    def __init__(self, folder_path, mode, method, visualize) -> None:
        
        self.folder_path = folder_path
        self.mode = mode
        self.method = method
        self.visualize = visualize
        
        self.sample_name = None
        self.sampling_rate_MHz = 30
        self.default_frame = 0
        
        self.R_channel_3d = None
        self.G_channel_3d = None
        self.B_channel_3d = None
        
        self.R_channel_2d = None
        self.G_channel_2d = None
        self.B_channel_2d = None

        self.df_trimmed_signal_3d = None
        self.df_trimmed_signal_envelope_3d = None
        self.df_convolved_signal_with_ghx_1_3d = None
        self.df_convolved_signal_with_ghx_1_envelope_3d = None
        self.df_convolved_signal_with_ghx_2_3d = None
        self.df_convolved_signal_with_ghx_2_envelope_3d = None
        
        self.df_trimmed_signal_2d = None
        self.df_trimmed_signal_envelope_2d = None
        self.df_convolved_signal_with_ghx_1_2d = None
        self.df_convolved_signal_with_ghx_1_envelope_2d = None
        self.df_convolved_signal_with_ghx_2_2d = None
        self.df_convolved_signal_with_ghx_2_envelope_2d = None
        
        self.df_trimmed_signal_1d = None
        self.df_trimmed_signal_envelope_1d = None
        self.df_convolved_signal_with_ghx_1_1d = None
        self.df_convolved_signal_with_ghx_1_envelope_1d = None
        self.df_convolved_signal_with_ghx_2_1d = None
        self.df_convolved_signal_with_ghx_2_envelope_1d = None
        
        self.R0B_trimmed_3x2d = None
        self.RGB_trimmed_3x2d = None

        self.__run()
       
    ###################################################################################

    def __run(self):
        
        # Read 3D trimmed signals
        self.df_trimmed_signal_3d = self.read_csv("trimmed_signal_3d.csv")
        self.df_trimmed_signal_envelope_3d = self.read_csv("trimmed_signal_envelope_3d.csv")

        # Read 3D convolved signals with GHX 1
        self.df_convolved_signal_with_ghx_1_3d = self.read_csv("convolved_signal_with_ghx_1_3d.csv")
        self.df_convolved_signal_with_ghx_1_envelope_3d = self.read_csv("convolved_signal_with_ghx_1_envelope_3d.csv")

        # Read 3D convolved signals with GHX 2
        self.df_convolved_signal_with_ghx_2_3d = self.read_csv("convolved_signal_with_ghx_2_3d.csv")
        self.df_convolved_signal_with_ghx_2_envelope_3d = self.read_csv("convolved_signal_with_ghx_2_envelope_3d.csv")
        
        # Extract 2D data from 3D trimmed signals
        self.df_trimmed_signal_2d = self.extract_data_for_frame(self.df_trimmed_signal_3d)
        self.df_trimmed_signal_envelope_2d = self.extract_data_for_frame(self.df_trimmed_signal_envelope_3d)

        # Extract 2D data from 3D convolved signals with GHX 1
        self.df_convolved_signal_with_ghx_1_2d = self.extract_data_for_frame(self.df_convolved_signal_with_ghx_1_3d)
        self.df_convolved_signal_with_ghx_1_envelope_2d = self.extract_data_for_frame(self.df_convolved_signal_with_ghx_1_envelope_3d)

        # Extract 2D data from 3D convolved signals with GHX 2
        self.df_convolved_signal_with_ghx_2_2d = self.extract_data_for_frame(self.df_convolved_signal_with_ghx_2_3d)
        self.df_convolved_signal_with_ghx_2_envelope_2d = self.extract_data_for_frame(self.df_convolved_signal_with_ghx_2_envelope_3d)

        # Extract middle signal as 1D from 2D DataFrames
        self.df_trimmed_signal_1d = self.extract_middle_signal_from_dataframe(self.df_trimmed_signal_2d)
        self.df_trimmed_signal_envelope_1d = self.extract_middle_signal_from_dataframe(self.df_trimmed_signal_envelope_2d)

        self.df_convolved_signal_with_ghx_1_1d = self.extract_middle_signal_from_dataframe(self.df_convolved_signal_with_ghx_1_2d)
        self.df_convolved_signal_with_ghx_1_envelope_1d = self.extract_middle_signal_from_dataframe(self.df_convolved_signal_with_ghx_1_envelope_2d)

        self.df_convolved_signal_with_ghx_2_1d = self.extract_middle_signal_from_dataframe(self.df_convolved_signal_with_ghx_2_2d)
        self.df_convolved_signal_with_ghx_2_envelope_1d = self.extract_middle_signal_from_dataframe(self.df_convolved_signal_with_ghx_2_envelope_2d)
        
        self.get_sample_name()
        
        # prepare imaging data
        self.create_color_channels_2d()
        self.prepare_R0B_3x2d()
        self.prepare_RGB_3x2d()

        self.visualization()     
        
    ###################################################################################
    
    
    
    # data
    ###################################################################################
    @staticmethod
    def normalize_with_min_max(original_array, new_min, new_max):
        min_value = np.min(original_array)
        max_value = np.max(original_array)
        normalized_array = ((original_array - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
        return normalized_array

    ###################################################################################
    @staticmethod
    def positive_negative_separator_xd(arr_xd, limit=0):
        positive_array = np.zeros_like(arr_xd)
        negative_array = np.zeros_like(arr_xd)

        # 2D array
        if len(arr_xd.shape) == 2:
            for i in range(arr_xd.shape[0]):
                for j in range(arr_xd.shape[1]):
                    if arr_xd[i, j] > limit:
                        positive_array[i, j] = arr_xd[i, j]
                    if arr_xd[i, j] < -limit:
                        negative_array[i, j] = arr_xd[i, j]
                        
        # 3D array
        elif len(arr_xd.shape) == 3:
            for k in range(arr_xd.shape[2]):
                for i in range(arr_xd.shape[0]):
                    for j in range(arr_xd.shape[1]):
                        if arr_xd[i, j, k] > limit:
                            positive_array[i, j, k] = arr_xd[i, j, k]
                        if arr_xd[i, j, k] < -limit:
                            negative_array[i, j, k] = arr_xd[i, j, k]

        return positive_array, negative_array
    
    ###################################################################################
    @staticmethod
    def shift_to_positive(arr_xd):
        
        # Find the minimum element in the array
        min_val = np.min(arr_xd)
        
        # Calculate the shift value to make all elements positive
        shift_val = abs(min_val) if min_val < 0 else 0
        
        # Shift all elements by the shift value
        shifted_arr = arr_xd + shift_val
        
        return shifted_arr
    
    ###################################################################################
    @staticmethod
    def normalize_to_0_1(
                   original_array: np.ndarray,   # Input numpy array to be normalized (np.ndarray)
                   ) -> np.ndarray:  # The maximum value of the new range (int)

        # Normalizing the values of the original array to the new range.
        normalized_array = original_array / np.max(original_array)

        return normalized_array
    
    ################################################################################### 
    
    def read_csv(self, file_name):
        # Construct the full file path
        file_path = os.path.join(self.folder_path, file_name)
        
        # Read the CSV file into a DataFrame
        try:
            logging.info("CSV file successfully read.")
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return None
        except pd.errors.EmptyDataError:
            logging.warning("The file is empty.")
            return None
        except pd.errors.ParserError:
            logging.error("Error parsing the file.")
            return None
        except Exception as e:
            logging.exception("An error occurred while reading the CSV file.")
            return None     
        
    ###################################################################################
    
    def get_sample_name(self):
        try:
            self.sample_name = os.path.basename(os.path.normpath(self.folder_path))
            logging.info(f'Sample name extracted: {self.sample_name}')
        except Exception as e:
            logging.error(f'Error extracting sample name: {e}')
            
    ###################################################################################

    def extract_data_for_frame(self, df):
        """Extracts data for a given frame index from the DataFrame."""
        if 'Frame_Index' not in df.columns:
            raise ValueError("The DataFrame must contain a 'Frame_Index' column.")
        
        data = df[df['Frame_Index'] == self.default_frame]
        data = data.drop(columns=['Frame_Index'])
        return data
    
    ###################################################################################

    def extract_middle_signal_from_dataframe(self, df):

        # Get the number of rows
        num_rows = len(df)
        
        # Calculate the middle index
        middle_index = num_rows // 2  # Integer division

        # Extract the middle row and convert it to a 1D array
        signal = df.iloc[middle_index].to_numpy()
        
        return signal
    
    ###################################################################################
    
    

    # hscan core methods
    ###################################################################################   

    def hscan_core_method_1(self,
                            red_xd,
                            green_xd,
                            blue_xd):

        # Normalize envelope for color channel and get log
        red_xd   = self.normalize_with_min_max(red_xd, 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
        
        # Log transformation
        red_xd = np.log10(red_xd)
        green_xd = np.log10(green_xd)
        blue_xd = np.log10(blue_xd)
                        
        # Difference calculation
        RB_difference_xd = red_xd - blue_xd
        positive, negative = self.positive_negative_separator_xd(arr_xd=RB_difference_xd, limit=0)
        
        # Assign values based on the difference
        red_xd = positive
        green_xd = green_xd
        blue_xd = np.abs(negative)
        
        return red_xd, green_xd, blue_xd
        
    ###################################################################################   

    def hscan_core_method_2(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # Normalize color channel 0-255
        red_xd   = self.normalize_with_min_max(red_xd , 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
        
        # log
        red_xd   = np.log10(red_xd)
        green_xd = np.log10(green_xd)
        blue_xd  = np.log10(blue_xd)
            
        return red_xd, green_xd, blue_xd
                
    ###################################################################################   
    
    def hscan_core_method_3(self,
                        red_xd,
                        green_xd,
                        blue_xd):
    
        # Normalize color channel 0-255
        red_xd   = self.normalize_with_min_max(red_xd , 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
                   
        return red_xd, green_xd, blue_xd
    
    ###################################################################################   

    def hscan_core_method_4(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # Normalize color channel 0-255
        red_xd   = self.normalize_with_min_max(red_xd , 1, 255)
        green_xd = self.normalize_with_min_max(green_xd, 1, 255)
        blue_xd  = self.normalize_with_min_max(blue_xd, 1, 255)
        
        # log
        red_xd   = np.log10(red_xd)
        green_xd = np.log10(green_xd)
        blue_xd  = np.log10(blue_xd)
            
        # shift to positive
        red_xd   = self.shift_to_positive(red_xd)
        green_xd = self.shift_to_positive(green_xd)
        blue_xd  = self.shift_to_positive(blue_xd)
    
        return red_xd, green_xd, blue_xd    
    
    ###################################################################################   

    def hscan_core_method_5(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # log
        red_xd   = 20 * np.log10(np.abs(1 + red_xd))
        green_xd = 20 * np.log10(np.abs(1 + green_xd))
        blue_xd  = 20 * np.log10(np.abs(1 + blue_xd))
        
        # shift to positive
        red_xd   = self.shift_to_positive(red_xd)
        green_xd = self.shift_to_positive(green_xd)
        blue_xd  = self.shift_to_positive(blue_xd)
        
        # Normalize color channel 0-1
        red_xd   = self.normalize_to_0_1(red_xd)
        green_xd = self.normalize_to_0_1(green_xd)
        blue_xd  = self.normalize_to_0_1(blue_xd)
        
        return red_xd, green_xd, blue_xd
                        
    ###################################################################################   

    def hscan_core_method_6(self,
                            red_xd,
                            green_xd,
                            blue_xd):
        
        # Normalize color channel 0-1
        red_xd   = self.normalize_to_0_1(red_xd)
        green_xd = self.normalize_to_0_1(green_xd)
        blue_xd  = self.normalize_to_0_1(blue_xd)
                    
        return red_xd, green_xd, blue_xd
        
    ###################################################################################   
            
            
                
    # color channel
    ###################################################################################   

    def create_color_channels_2d(self):
        
        if self.method == "method_1":
            # Implement logic for method_2
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_1(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_2":
            # Implement logic for method_2
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_2(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_3":
            # Implement logic for method_3
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_3(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_4":
            # Implement logic for method_4
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_4(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )

        elif self.method == "method_5":
            # Implement logic for method_5
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_5(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )
            
        elif self.method == "method_6":
            # Implement logic for method_5
            self.R_channel_2d, self.G_channel_2d, self.B_channel_2d = self.hscan_core_method_6(
                red_xd=self.df_convolved_signal_with_ghx_1_envelope_2d.values,
                green_xd=self.df_trimmed_signal_envelope_2d.values,
                blue_xd=self.df_convolved_signal_with_ghx_2_envelope_2d.values
            )
            
    ###################################################################################   

    def prepare_R0B_3x2d(self):
        
        # create R0B channel with mask
        self.R0B_trimmed_3x2d = np.stack([self.R_channel_2d,
                                            np.zeros_like(self.G_channel_2d),
                                            self.B_channel_2d], axis=-1)    
        
    ###################################################################################

    def prepare_RGB_3x2d(self):

        # full RGB image
        self.RGB_trimmed_3x2d = np.stack([self.R_channel_2d,
                                            self.G_channel_2d,
                                            self.B_channel_2d], axis=-1)

    ###################################################################################
    
    
    
    # visualize
    ###################################################################################
    @staticmethod
    def rotate_flip(
                    two_dimension_array: np.ndarray) -> np.ndarray:

        # Rotate the input array counterclockwise by 90 degrees
        rotated_array = np.rot90(two_dimension_array)
        
        # Flip the rotated array horizontally
        rotated_flipped_array = np.flipud(rotated_array)
        
        return rotated_flipped_array
    
    ###################################################################################
    
    def image_envelope(self,
                    signal_2d,
                    log=False):

        # Create a new figure
        plt.rcdefaults()
        plt.figure(figsize=(8, 6))

        # Apply logarithmic transformation
        if log:
            signal_2d = 20 * np.log10(np.abs(1 + signal_2d))
        
        # Adjust the image: Rotate and flip
        signal_2d = self.rotate_flip(signal_2d) # Flip upside down and rotate 90 degrees

        # Display the image with the specified colormap
        plt.imshow(signal_2d, cmap='gray', aspect='auto')
        plt.title(f'Envelope image, {self.sample_name}, log scale')  

        # Add a color bar       
        plt.colorbar(label='dB')

        # Adjust layout for better display
        plt.tight_layout()

        # Show the plot
        plt.show()

    ###################################################################################

    def image_RB_difference_in_RGB_2d(self,
                                        R0B_final_2d: np.ndarray,  # RB final image as a numpy array
                                        additional_text: str = "",
                                        log: bool = False,  # Set default to False
                                        ) -> None:  # Default filename

        # Define custom color points for the colormap
        colors = [(1, 0, 0), (0, 0, 0), (0, 0, 1)]  # Red, Black, Blue
        
        # Create a custom colormap
        cmap_name = 'my_custom_colorbar'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        
        # Create a new figure
        plt.figure(figsize=(8, 6))
        
        if log:
            R0B_final_2d = 20 * np.log10(np.abs(1 + R0B_final_2d))

        R0B_final_2d = self.rotate_flip(R0B_final_2d)
        
        # Display the image
        plt.imshow(R0B_final_2d, aspect='auto', cmap=custom_cmap)

        plt.title(f'RB image zero G in RGB, {additional_text}')
        
        # Add a color bar with dynamic ticks
        cbar = plt.colorbar()
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Large', 'Medium', 'Small'])

        # Adjust layout for better display
        plt.tight_layout()
        plt.show()
  
    ###################################################################################
    
    def image_color_channel_2d(self,
                                color_channel_2d: np.ndarray,   # Red channel without RGB as a numpy array
                                color = None,
                                addtional_text = None,
                                log: bool = None, 
                                ) -> None:         # Flag indicating whether to save the image

        # Create a new figure
        plt.figure(figsize=(8, 6))
        
        if log:
            
            color_channel_2d = 20 * np.log10(np.abs(1 + color_channel_2d))

            if color == "red":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Reds')
            elif color == "green":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Greens')
            elif color == "blue":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Blues')
            elif color == "RGB":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto')
                #plt.colorbar().remove()
        
        else:    
            
            if color == "red":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Reds')
            elif color == "green":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Greens')
            elif color == "blue":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto', cmap='Blues')
            elif color == "RGB":
                plt.imshow(self.rotate_flip(color_channel_2d), aspect='auto')
                #plt.colorbar().remove()
        
        # Add a color bar
        plt.colorbar()

        # Set the title of the plot
        plt.title(f'{addtional_text}')  
        
        # Adjust layout for better display
        plt.tight_layout()

        plt.show()

    ###################################################################################

    def plot_1d_signal_envelope_and_fft(self, signal, envelope):
        # Generate time axis based on the length of the signal
        num_samples = len(signal)
        time_microseconds = np.arange(num_samples)  # Use sample indices as time in microseconds

        # Calculate FFT
        fft_signal = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_signal)

        # Frequency axis in MHz using the sampling frequency
        freq_MHz = np.fft.fftfreq(num_samples, d=(1 / (self.sampling_rate_MHz)))  # Frequency in Hz

        # Only take positive frequencies
        positive_freq_indices = freq_MHz >= 0
        freq_positive = freq_MHz[positive_freq_indices]
        fft_magnitude_positive = fft_magnitude[positive_freq_indices]

        # Create figure
        plt.figure(figsize=(14, 4))

        # Plot the original signal and its envelope
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
        plt.plot(time_microseconds, signal, color='blue', label='Signal')  # Original signal
        plt.plot(time_microseconds, envelope, color='orange', linestyle='--', label='Envelope')  # Envelope
        plt.title('1D Signal Plot')
        plt.xlabel('samples')  # Label for microseconds
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # Plot the positive FFT
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
        plt.plot(freq_positive, fft_magnitude_positive, color='green')  # Frequency in MHz
        plt.title('FFT of Signal')
        plt.xlabel('Frequency (MHz)')  # Label for MHz
        plt.ylabel('Magnitude')
        plt.grid(True)

        plt.tight_layout()  # Adjust layout
        plt.show()
        
    ###################################################################################

    def plot_envelope(self, envelope, color):
        # Generate time axis based on the length of the envelope
        num_samples = len(envelope)
        time_microseconds = np.arange(num_samples)  # Use sample indices as time in microseconds

        # Create figure
        plt.figure(figsize=(7, 4))

        # Plot the envelope with the specified color
        plt.plot(time_microseconds, envelope, color=color, linestyle='--', label='Envelope')  # Envelope
        plt.title('Envelope of the Signal')
        plt.xlabel('Samples')  # Label for microseconds
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()  # Adjust layout
        plt.show()

    ###################################################################################

    def visualization(self):
        
        if self.visualize:
            
            self.image_envelope(signal_2d = self.df_trimmed_signal_envelope_2d, log=True)
            
            self.image_RB_difference_in_RGB_2d(R0B_final_2d = self.R0B_trimmed_3x2d, 
                                                additional_text="",
                                                log=False)
            
            self.image_color_channel_2d(color_channel_2d = self.RGB_trimmed_3x2d, color="RGB", addtional_text="RGB_based on first frame", log=False)
            self.image_color_channel_2d(color_channel_2d = self.G_channel_2d, color="green", addtional_text="single channel", log=False)
            self.image_color_channel_2d(color_channel_2d = self.R_channel_2d, color="red",   addtional_text="single channel", log=False)
            self.image_color_channel_2d(color_channel_2d = self.B_channel_2d, color="blue",  addtional_text="single channel", log=False)

            # plot raw signal with wavelets
            self.plot_1d_signal_envelope_and_fft(signal = self.df_trimmed_signal_1d,
                                        envelope = self.df_trimmed_signal_envelope_1d)
            
            self.plot_1d_signal_envelope_and_fft(signal = self.df_convolved_signal_with_ghx_1_1d,
                                        envelope = self.df_convolved_signal_with_ghx_1_envelope_1d)
                    
            self.plot_1d_signal_envelope_and_fft(signal = self.df_convolved_signal_with_ghx_2_1d,
                                        envelope = self.df_convolved_signal_with_ghx_2_envelope_1d)
            
            self.plot_envelope(self.G_channel_2d[self.G_channel_2d.shape[0]//2, :], color = 'green')
            self.plot_envelope(self.R_channel_2d[self.G_channel_2d.shape[0]//2, :], color = 'red')
            self.plot_envelope(self.B_channel_2d[self.G_channel_2d.shape[0]//2, :], color = 'blue')

    ###################################################################################







