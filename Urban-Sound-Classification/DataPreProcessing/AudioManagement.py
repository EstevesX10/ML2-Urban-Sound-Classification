import librosa as libr
import numpy as np
import pandas as pd

def formatFilePath(audioFold:int, audioName:str) -> str:
    """
    # Description
        -> Creates a filepath to correctly access a audio file from the UrbanSound8K dataset.
    -----------------------------------------------------------------------------------------
    := param: audioFold - Fold where the audio sample belong to inside the dataset.
    := param: audioName - Audio Filename inside the dataset.
    := return: String that points to the correct file.
    """

    # Return the file path
    return f'./UrbanSound8K/audio/fold{audioFold}/{audioName}'

def loadAudio(audioSliceName:int, audioDuration:int, df:pd.DataFrame) -> np.ndarray:
    """
    # Description
        -> Loads a audio file from the dataset.
    -------------------------------------------
    := param: audioSliceName - Audio Identification inside the dataset.
    := param: audioDuration - Duration to be considered of the audio.
    := param: df - Pandas DataFrame with the dataset's metadata.
    := return: Audio object.
    """
    
    # Get the audio entry
    df_selectedAudio = df[df['slice_file_name'] == audioSliceName]

    # Get the row index of the entry
    idx = df_selectedAudio.index.values.astype(int)[0]

    # Fetch audio fold
    audioFold = df_selectedAudio['fold'][idx]
    
    # Format the File Path
    audioFilePath = formatFilePath(audioFold, audioSliceName)
    
    # Load the audio
    audioTimeSeries, samplingRate = libr.load(audioFilePath, duration=audioDuration)

    # Return the Audio
    return audioTimeSeries, samplingRate