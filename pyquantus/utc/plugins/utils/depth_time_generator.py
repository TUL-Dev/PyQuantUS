import numpy as np

###################################################################################
# Depth and time array generator
###################################################################################
class DepthTimeArrayGenerator:
    """Class for generating depth and time arrays for the ROI.
    
    This class generates arrays for depth and time values based on signal parameters.
    The depth array represents the penetration depth in centimeters, while the time
    array represents the corresponding time values in seconds.
    
    Attributes:
        signal_length (int): Length of the signal in samples
        sampling_frequency (float): Sampling frequency in Hz
        speed_of_sound (float): Speed of sound in m/s
        penetration_depth (float): Maximum penetration depth in cm
        depth_array (np.ndarray): Array of depth values in cm
        time_array (np.ndarray): Array of time values in seconds
    """
    def __init__(self,
                 signal_len: int,
                 sampling_frequency_MHz: float,
                 speed_of_sound_m_s: float):
        
        # input arguments
        self.signal_len = signal_len
        self.sampling_frequency_MHz = sampling_frequency_MHz
        self.speed_of_sound_m_s = speed_of_sound_m_s
        
        # computed attributes
        self.depth_array_cm = None
        self.time_array_s = None
        
        self.__run()
        
    ###################################################################################
    # Run
    ###################################################################################
    def __run(self):
        
        self.set_time_array()
        self.set_depth_array()
        
    ###################################################################################
    # Set time array
    ###################################################################################
    def set_time_array(self):
        self.time_array_s = np.arange(self.signal_len) / self.sampling_frequency_MHz * 1e-6
        
    ###################################################################################
    # Set depth array
    ###################################################################################
    def set_depth_array(self):
        self.depth_array_cm = (self.speed_of_sound_m_s * self.time_array_s / 2) * 100  # Convert to cm
        
     