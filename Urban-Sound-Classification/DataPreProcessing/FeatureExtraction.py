import numpy as np
import pandas as pd
import os
import librosa
from .AudioManagement import (loadAudio)

def extract1DimensionalData(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract 1-Dimensional features 
        from the audio samples of the selected Fold on the dataset
    --------------------------------------------------------------
    := param: audio_df - Pandas Dataframe with the UrbamSound8K dataset metadata.
    := param: fold - Fold of the audios that we want to perform feature extraction on.
    := param: config - Dictionary with Constants used in audio processing throughout the project.
    := param: pathsConfig - Dictionary with filepaths to help organize the results of the project.
    := return: None, since we are merely extracting data.
    """

    # Check if the datframe has already been computed
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['1-Dimensional-Data']):
        # Initialize a List to store the extracted content
        data = []

        # Get the audio filenames from the selected fold
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Iterate through all the audios inside the selected fold
        for audioFileName in foldAudios:
            # Load the Audio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # Compute and append the extracted features to the data list
            data.append({
                'Audio':audioFileName,
                'Zero-Crossing Rate':librosa.feature.zero_crossing_rate(y=audio),
                'Spectral Centroid':librosa.feature.spectral_centroid(y=audio, sr=config['SAMPLE_RATE']),
                'Spectral Bandwidth':librosa.feature.spectral_bandwidth(y=audio, sr=config['SAMPLE_RATE']),
                'Spectral Flatness':librosa.feature.spectral_flatness(y=audio),
                'Spectral Roll-off':librosa.feature.spectral_rolloff(y=audio, sr=config['SAMPLE_RATE']),
                'RMS Energy':librosa.feature.rms(y=audio)
            })

        # Create a DataFrame with the collected data
        df = pd.DataFrame(data)

        # Save the Dataframe
        df.to_csv(pathsConfig['Datasets'][f'Fold-{fold}']['1-Dimensional-Data'], sep=',', index=False)
    else:
        print(f"[Fold-{fold}] 1-Dimensional Features have already been Extracted!")

def extract2DimensionalData(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract 2-Dimensional features 
        from the audio samples of the selected Fold on the dataset
    --------------------------------------------------------------
    := param: audio_df - Pandas Dataframe with the UrbamSound8K dataset metadata.
    := param: fold - Fold of the audios that we want to perform feature extraction on.
    := param: config - Dictionary with Constants used in audio processing throughout the project.
    := param: pathsConfig - Dictionary with filepaths to help organize the results of the project.
    := return: None, since we are merely extracting data.
    """

    # Check if the datframe has already been computed
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['2-Dimensional-Data']):
        # Initialize a List to store the extracted content
        data = []

        # Get the audio filenames from the selected fold
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Iterate through all the audios inside the selected fold
        for audioFileName in foldAudios:
            # Load the Audio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # Compute and append the extracted features to the data list
            data.append({
                'Audio':audioFileName,
                'MFCC':librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC']),
                'Chroma STFT':librosa.feature.chroma_stft(y=audio, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH']),
                'Mel Spectrogram':librosa.feature.melspectrogram(y=audio, sr=config['SAMPLE_RATE']),
                'Spectral Contrast':librosa.feature.spectral_contrast(y=audio, sr=config['SAMPLE_RATE'])
            })

        # Create a DataFrame with the collected data
        df = pd.DataFrame(data)

        # Save the Dataframe
        df.to_csv(pathsConfig['Datasets'][f'Fold-{fold}']['2-Dimensional-Data'], sep=',', index=False)
    else:
        print(f"[Fold-{fold}] 2-Dimensional Features have already been Extracted!")