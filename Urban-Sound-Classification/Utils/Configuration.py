def loadConfig() -> dict:
    """
    # Description
        -> This function aims to store all the configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important constants/values used in the project.
    """

    # Computing Values
    sampleRate = 44100 # Higher rate to be able to capture high resolution audios like the ones that come from harmonic waves.
    hopLength = round(sampleRate * 0.0125)
    windowLength = round(sampleRate * 0.023)
    timeSize = (4 * sampleRate // hopLength + 1)
    return {
        'DURATION':4,                   # Length of each audio sample in the dataset.
        'SAMPLE_RATE':sampleRate,       # Number of samples of audio taken per second when converting it from a continuous to a digital signal
        'HOP_LENGTH':hopLength,         # The number of samples to advance between frames
        'WINDOW_LENGTH':windowLength,   # Number of samples used in each frame for frequency analysis, or the length of the window in which the Fourier Transform is applied.
        'N_FFT':2**10,                  # Length of the windowed signal after padding with zeros
        'TIME_SIZE':timeSize,           # Number of time frames or segments that the audio will be divided into after applying the hop length and windowing.
        'N_CHROMA':12,                  # Number of pitch classes (e.g., C, C#, D, etc.) in the chroma feature representation.
        'N_MFCC':13                     # Number of Mel-Frequency Cepstral Coefficients (MFCCs) to be extracted
    }

def loadPathsConfig() -> dict:
    """
    # Description
        -> This function aims to store all the path configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important file paths of the project.
    """
    return {
        'ExploratoryDataAnalysis': './ExperimentalResults/ExploratoryDataAnalysis',
        'ModelDevelopmentAndEvaluation': {
            'MLP': './ExperimentalResults/ModelDevelopmentAndEvaluation/MLP/',
            'CNN': './ExperimentalResults/ModelDevelopmentAndEvaluation/CNN/',
            'RNN': './ExperimentalResults/ModelDevelopmentAndEvaluation/RNN/' # Maybe change later
        }
    }