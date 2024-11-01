import librosa as libr
import numpy as np
import pandas as pd
from IPython import display

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

def loadAudio(df_audio:pd.DataFrame, audioSliceName:int, audioDuration:int, targetSampleRate:int, usePadding:bool) -> np.ndarray:
    """
    # Description
        -> Loads a audio file from the dataset.
    -------------------------------------------
    := param: df_audio - Pandas DataFrame with the dataset's metadata.
    := param: audioSliceName - Audio Identification inside the dataset.
    := param: audioDuration - Duration to be considered of the audio.
    := param: targetSampleRate - Target sampling rate for the audio.
    := param: usePadding - Whether or not to perform zero padding to the resampled audio on the target sample rate.
    := return: Audio object.
    """
    
    # Get the audio entry
    df_audio_selectedAudio = df_audio[df_audio['slice_file_name'] == audioSliceName]

    # Get the row index of the entry
    idx = df_audio_selectedAudio.index.values.astype(int)[0]

    # Fetch audio fold
    audioFold = df_audio_selectedAudio['fold'][idx]
    
    # Format the File Path
    audioFilePath = formatFilePath(audioFold, audioSliceName)
    
    # Load the audio [Using standard sampling rate]
    audioTimeSeries, samplingRate = libr.load(audioFilePath, duration=audioDuration, sr=None)

    # Resample the audio for the target sample rate
    audioTimeSeries = libr.resample(audioTimeSeries, orig_sr=samplingRate, target_sr=targetSampleRate)

    # Perform padding on the audio, so that each time series have the same length
    if usePadding:
        audioTimeSeries = libr.util.fix_length(data=audioTimeSeries, size=audioDuration*targetSampleRate, mode='constant')

    # Return the padded Audio
    return audioTimeSeries

def showcaseAudio(df_audio:pd.DataFrame, audioSliceName:int) -> display.Audio:
    """
    # Description
        -> Creates a simple audio player for the selected file from the dataset.
    ----------------------------------------------------------------------------
    := param: df_audio - Pandas DataFrame with the dataset's metadata.
    := param: audioSliceName - Audio Identification inside the dataset.
    := return: Audio object that helps listen to the selected audio file.
    """
    
    # Get the audio entry
    df_audio_selectedAudio = df_audio[df_audio['slice_file_name'] == audioSliceName]

    # Get the row index of the entry
    idx = df_audio_selectedAudio.index.values.astype(int)[0]

    # Fetch audio fold
    audioFold = df_audio_selectedAudio['fold'][idx]

    # Format the File Path
    audioFilePath = formatFilePath(audioFold, audioSliceName)

    # Return the Audio
    return display.Audio(audioFilePath)