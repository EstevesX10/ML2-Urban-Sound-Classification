def loadConfig() -> dict:
    """
    # Description
        -> This function aims to store all the configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important constants/values used in the project.
    """

    # Computing Values
    samplingRate = 22050
    hopLength = round(samplingRate * 0.0125)
    windowLength = round(samplingRate * 0.023)
    timeSize = ((4*samplingRate) // (hopLength + 1))
    return {
        'DURATION':4,                   # Duration of the Audio
        'SAMPLING_RATE':samplingRate,   # Number of samples of audio taken per second when converting it from a continuous to a digital signal
        'HOP_LENGTH':hopLength,         # The number of samples to advance between frames
        'WINDOW_LENGTH':windowLength,   #
        'N_FFT':2**10,                  # Length of the windowed signal after padding with zeros
        'TIME_SIZE':timeSize,           #
        'N_CHROMA':12                   #
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