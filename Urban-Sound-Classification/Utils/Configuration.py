def loadConfig() -> dict:
    """
    # Description
        -> This function aims to store all the configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important file paths of the project.
    """

    # Computing Values
    sampleRate = 22050
    hopLength = round(sampleRate * 0.0125)
    windowLength = round(sampleRate * 0.023)
    timeSize = ((4*sampleRate) // (hopLength + 1))
    return {
        'DURATION':4,
        'SAMPLE_RATE':sampleRate,
        'HOP_LENGTH':hopLength,
        'WINDOW_LENGTH':windowLength,
        'N_FFT':2**10,
        'TIME_SIZE':timeSize
    }