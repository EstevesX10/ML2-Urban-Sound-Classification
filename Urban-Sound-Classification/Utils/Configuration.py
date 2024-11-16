def loadConfig() -> dict:
    """
    # Description
        -> This function aims to store all the configuration related parameters used inside the project.
    ----------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important constants/values used in the project.
    """

    # Computing Values
    sampleRate = 44100  # Higher rate to be able to capture high resolution audios like the ones that come from harmonic waves.
    hopLength = round(sampleRate * 0.0125)
    windowLength = round(sampleRate * 0.023)
    timeSize = 4 * sampleRate // hopLength + 1
    return {
        "DURATION": 4,  # Length of each audio sample in the dataset.
        "SAMPLE_RATE": sampleRate,  # Number of samples of audio taken per second when converting it from a continuous to a digital signal
        "HOP_LENGTH": hopLength,  # The number of samples to advance between frames
        "WINDOW_LENGTH": windowLength,  # Number of samples used in each frame for frequency analysis, or the length of the window in which the Fourier Transform is applied.
        "N_FFT": 2**10,  # Length of the windowed signal after padding with zeros
        "TIME_SIZE": timeSize,  # Number of time frames or segments that the audio will be divided into after applying the hop length and windowing.
        "N_CHROMA": 12,  # Number of pitch classes (e.g., C, C#, D, etc.) in the chroma feature representation.
        "N_MFCC": 13,  # Number of Mel-Frequency Cepstral Coefficients (MFCCs) to be extracted
    }

def loadPathsConfig() -> dict:
    """
    # Description
        -> This function aims to store all the path configuration related parameters used inside the project.
    ---------------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important file paths of the project.
    """
    return {
        "ExploratoryDataAnalysis": "./ExperimentalResults/ExploratoryDataAnalysis",
        "Datasets": {
            "Fold-1": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-1/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-1/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-1/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-1/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-1/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-1/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-1/1D-Processed-MFCCs.pkl",
            },
            "Fold-2": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-2/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-2/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-2/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-2/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-2/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-2/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-2/1D-Processed-MFCCs.pkl",
            },
            "Fold-3": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-3/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-3/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-3/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-3/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-3/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-3/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-3/1D-Processed-MFCCs.pkl",
            },
            "Fold-4": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-4/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-4/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-4/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-4/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-4/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-4/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-4/1D-Processed-MFCCs.pkl",
            },
            "Fold-5": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-5/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-5/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-5/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-5/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-5/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-5/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-5/1D-Processed-MFCCs.pkl",
            },
            "Fold-6": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-6/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-6/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-6/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-6/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-6/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-6/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-6/1D-Processed-MFCCs.pkl",
            },
            "Fold-7": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-7/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-7/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-7/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-7/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-7/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-7/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-7/1D-Processed-MFCCs.pkl",
            },
            "Fold-8": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-8/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-8/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-8/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-8/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-8/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-8/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-8/1D-Processed-MFCCs.pkl",
            },
            "Fold-9": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-9/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-9/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-9/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-9/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-9/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-9/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-9/1D-Processed-MFCCs.pkl",
            },
            "Fold-10": {
                # All the Available Features to process the Audio Samples
                "All-Raw-Features": "./Datasets/Fold-10/All-Raw-Features.pkl",

                # Files to store 1-Dimensional Features
                "1D-Raw-Features": "./Datasets/Fold-10/1D-Raw-Features.pkl",
                "1D-Processed-Features": "./Datasets/Fold-10/1D-Processed-Features.pkl",
                
                # Files to store 2-Dimensional Features
                "2D-Raw-Features": "./Datasets/Fold-10/2D-Raw-Features.pkl",
                "2D-Processed-Features": "./Datasets/Fold-10/2D-Processed-Features.pkl",

                # Files to store the MFCCs 
                "2D-Raw-MFCCs": "./Datasets/Fold-10/2D-Raw-MFCCs.pkl",
                "1D-Processed-MFCCs": "./Datasets/Fold-10/1D-Processed-MFCCs.pkl",
            },
        },
        "ModelDevelopmentAndEvaluation": {
            "MLP": "./ExperimentalResults/ModelDevelopmentAndEvaluation/MLP/",
            "CNN": "./ExperimentalResults/ModelDevelopmentAndEvaluation/CNN/",
            "yamnet-train": "./ExperimentalResults/ModelDevelopmentAndEvaluation/yamnet/train.pkl",
            "yamnet-test": "./ExperimentalResults/ModelDevelopmentAndEvaluation/yamnet/test.pkl",
        },
    }
