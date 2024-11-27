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


def createDatasetsFolderPaths(numberFolds: int) -> dict:
    """
    # Description
        -> This function helps configure all the paths in which we are going to store the extracted features.
    ---------------------------------------------------------------------------------------------------------
    := param: numberFolds -  Number of folds that are going to be considered.
    := return: Dictionary with all the datasets paths configured.
    """

    # Create a main dictionary to store all the paths
    datasetsPaths = {}

    for foldIdx in range(1, numberFolds + 1):
        # Define the paths for the current fold
        currentFoldDatasetsPaths = {
            # All the Available Features to process the Audio Samples
            "All-Raw-Features": f"./Datasets/Fold-{foldIdx}/All-Raw-Features.pkl",
            # Files to store 1-Dimensional Features
            "1D-Raw-Features": f"./Datasets/Fold-{foldIdx}/1D-Raw-Features.pkl",
            "1D-Processed-Features": f"./Datasets/Fold-{foldIdx}/1D-Processed-Features.pkl",
            # Files to store 2-Dimensional Features
            "2D-Raw-Features": f"./Datasets/Fold-{foldIdx}/2D-Raw-Features.pkl",
            "2D-Processed-Features": f"./Datasets/Fold-{foldIdx}/2D-Processed-Features.pkl",
            # Files to store the MFCCs
            "2D-Raw-MFCCs": f"./Datasets/Fold-{foldIdx}/2D-Raw-MFCCs.pkl",
            "1D-Processed-MFCCs": f"./Datasets/Fold-{foldIdx}/1D-Processed-MFCCs.pkl",
        }

        # Update the main dictionary
        datasetsPaths.update({f"Fold-{foldIdx}": currentFoldDatasetsPaths})

    # Add the path for transfer learning
    datasetsPaths.update({"transfer": "./Datasets/transfer.pkl"})

    # Return the dictionary with all the paths for the datasets
    return datasetsPaths


def createModelResultsFolderPaths(
    modelName: str, numberTests: int, numberFolds: int
) -> dict:
    """ "
    # Description
        -> This function helps create all the paths to store the experimental results of a given model.
    ---------------------------------------------------------------------------------------------------
    := param: modelName - Name of the model.
    := param: numberTests - Number of tests that are going to be performed on the model.
    := param: numberFolds -  Number of folds that are going to be considered.
    := return: Dictionary wtith all the proper paths formatted for the model's experimental results.
    """

    # Create a main dictionary to store all the paths for the results of a model
    testsData = {}

    # Iterate through the amount of tests to perform
    for testIdx in range(1, numberTests + 1):
        # Create a dictionary for the paths of the results from each fold for the current test
        foldsData = {}

        # Iterate through each fold
        for foldIdx in range(1, numberFolds + 1):
            # Create the paths
            foldsData.update(
                {
                    f"Fold-{foldIdx}": {
                        "History": f"./ExperimentalResults/ModelPerformanceResults/{modelName}/Test-{testIdx}/Fold-{foldIdx}/history.pkl",
                        "Model": f"./ExperimentalResults/ModelPerformanceResults/{modelName}/Test-{testIdx}/Fold-{foldIdx}/model.keras",
                    }
                }
            )
        # Update the main dictionary
        testsData.update({f"Test-{testIdx}": foldsData})

    # Return the Paths for the model experimental results
    return testsData


def loadPathsConfig() -> dict:
    """
    # Description
        -> This function aims to store all the path configuration related parameters used inside the project.
    ---------------------------------------------------------------------------------------------------------
    := return: Dictionary with some of the important file paths of the project.
    """
    return {
        "ExploratoryDataAnalysis": "./ExperimentalResults/ExploratoryDataAnalysis",
        "Datasets": createDatasetsFolderPaths(numberFolds=10),
        "ModelDevelopmentAndEvaluation": {
            "MLP": createModelResultsFolderPaths(
                modelName="MLP", numberTests=3, numberFolds=10
            ),
            "CNN": createModelResultsFolderPaths(
                modelName="CNN", numberTests=4, numberFolds=10
            ),
            "YAMNET": createModelResultsFolderPaths(
                modelName="YAMNET", numberTests=4, numberFolds=10
            ),
            "ResNet": createModelResultsFolderPaths(
                modelName="ResNet", numberTests=3, numberFolds=10
            ),
            "yamnet-train": "./ExperimentalResults/ModelDevelopmentAndEvaluation/yamnet/train.pkl",
            "yamnet-test": "./ExperimentalResults/ModelDevelopmentAndEvaluation/yamnet/test.pkl",
        },
    }
