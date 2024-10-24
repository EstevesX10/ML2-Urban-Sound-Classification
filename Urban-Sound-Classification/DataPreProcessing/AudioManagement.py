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

def loadAudio(audioID:int, df:pd.DataFrame) -> np.ndarray:
    """
    # Description
        -> Loads a audio file from the dataset.
    -------------------------------------------
    := param: audioID - Audio Identification Number inside the dataset.
    := param: df - Pandas DataFrame with the dataset's metadata.
    := return: Audio object.
    """
    
    # Get the audio entry
    df_selectedAudio = df.loc[df['fsID'] == audioID]

    # Get audio name
    audioName = df_selectedAudio['slice_file_name'][0]

    # Fetch audio fold
    audioFold = df_selectedAudio['fold'][0]
    
    # Get Start and End Time
    startTime = df_selectedAudio['start'][0]
    endTime = df_selectedAudio['end'][0]

    # Compute audio duration
    audioDuration = endTime - startTime
    
    # Format the File Path
    audioFilePath = formatFilePath(audioFold, audioName)
    
    # Load the audio
    audioTimeSeries, samplingRate = libr.load(audioFilePath, offset=startTime, duration=audioDuration)

    # Return the Audio
    return audioTimeSeries, samplingRate