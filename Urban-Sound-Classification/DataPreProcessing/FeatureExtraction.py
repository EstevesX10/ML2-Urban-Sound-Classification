import numpy as np
import pandas as pd
import os
import librosa
from sklearn.preprocessing import (MinMaxScaler)
from .AudioManagement import (loadAudio)

def extractMFCCs(audio_df:pd.DataFrame, raw:bool, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract the mean MFCCs from audio samples 
        of the selected Fold on the dataset.
    --------------------------------------------------------------------
    := param: audio_df - Pandas Dataframe with the UrbamSound8K dataset metadata.
    := param: raw - Boolean Value that determines whether or not we are to work with raw data.
    := param: fold - Fold of the audios that we want to perform feature extraction on.
    := param: config - Dictionary with Constants used in audio processing throughout the project.
    := param: pathsConfig - Dictionary with filepaths to help organize the results of the project.
    := return: None, since we are merely extracting data.
    """
    
    # Define a Default Value for the raw boolean
    raw = False if raw is None else raw

    # Define file path
    if raw:
        mfccsFilePath = pathsConfig['Datasets'][f'Fold-{fold}']['Raw-MFCCs-Feature']
    else:
        mfccsFilePath = pathsConfig['Datasets'][f'Fold-{fold}']['Processed-MFCCs-Feature']

    # Check if the dataframe has already been computed
    if not os.path.exists(mfccsFilePath):
        # Initialize a List to store the extracted content
        data = []

        # Get the audio filenames from the selected fold
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Iterate through all the audios inside the selected fold
        for audioFileName in foldAudios:
            # Load the Audio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # Compute the mfccs
            if raw: # Raw Data
                mfcc = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC']).tolist()
            else: # Mean on each coefficient
                mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC']), axis=1).tolist()

            # Compute and append the MFCC's
            data.append({
                # Audio Details
                'audio':audioFileName,
                'fold':fold,

                # MFCC
                'MFCC':mfcc,
                
                # Target
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

        # Create a DataFrame with the collected data
        df = pd.DataFrame(data)

        # Save the Dataframe
        df.to_csv(mfccsFilePath, sep=',', index=False)

    else:
        if raw:
            print(f"[Fold-{fold}]\tRaw MFCC's have already been Extracted!")
        else:
            print(f"[Fold-{fold}]\tProcessed MFCC's have already been Extracted!")
        
def extractAllFeatures(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract all the available features 
        from the audio samples of the selected Fold on the dataset
    --------------------------------------------------------------
    := param: audio_df - Pandas Dataframe with the UrbamSound8K dataset metadata.
    := param: fold - Fold of the audios that we want to perform feature extraction on.
    := param: config - Dictionary with Constants used in audio processing throughout the project.
    := param: pathsConfig - Dictionary with filepaths to help organize the results of the project.
    := return: None, since we are merely extracting data.
    """

    # Check if the dataframe has already been computed
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['Total-Features']):
        # Initialize a List to store the extracted content
        data = []

        # Get the audio filenames from the selected fold
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Iterate through all the audios inside the selected fold
        for audioFileName in foldAudios:
            # Load the Audio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # [Compute Features]

            # [1-Dimensional Features]
            # Zero Crossing Rate
            zeroCrossingRate = librosa.feature.zero_crossing_rate(y=audio).tolist()[0]

            # Spectral Centroid
            spectralCentroid = librosa.feature.spectral_centroid(y=audio, sr=config['SAMPLE_RATE']).tolist()[0]

            # Spectral Bandwidth
            spectralBandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=config['SAMPLE_RATE']).tolist()[0]

            # Spectral Flatness
            spectralFlatness = librosa.feature.spectral_flatness(y=audio).tolist()[0]

            # Spectral Roll-off
            spectralRolloff = librosa.feature.spectral_rolloff(y=audio, sr=config['SAMPLE_RATE']).tolist()[0]

            # RMS Energy
            rmsEnergy = librosa.feature.rms(y=audio).tolist()[0]

            # [2-Dimensional Features]
            # MFCCs
            mfccRaw = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])
            mfcc = mfccRaw.tolist()
            
            # Chroma STFT
            chromaSTFTRaw = librosa.feature.chroma_stft(y=audio, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])
            chromaSTFT = chromaSTFTRaw.tolist()

            # Mel Spectrogram
            melSpectrogramRaw = librosa.feature.melspectrogram(y=audio, sr=config['SAMPLE_RATE'])
            melSpectrogram = melSpectrogramRaw.tolist()

            # Spectral Contrast
            spectralContrastRaw = librosa.feature.spectral_contrast(y=audio, sr=config['SAMPLE_RATE'])
            spectralContrast = spectralContrastRaw.tolist()

            # Compute and append the extracted features to the data list
            data.append({
                # Audio Details
                'audio':audioFileName,
                'fold':fold,

                # 1-Dimensional Features
                'Zero-Crossing Rate':zeroCrossingRate,
                'Spectral Centroid':spectralCentroid,
                'Spectral Bandwidth':spectralBandwidth,
                'Spectral Flatness':spectralFlatness,
                'Spectral Roll-off':spectralRolloff,
                'RMS Energy':rmsEnergy,
                
                # 2-Dimensional Features
                'MFCC':mfcc,
                'Chroma STFT':chromaSTFT,
                'Mel Spectrogram':melSpectrogram,
                'Spectral Contrast':spectralContrast,

                # Target
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

        # Create a DataFrame with the collected data
        df = pd.DataFrame(data)

        # Save the Dataframe
        df.to_csv(pathsConfig['Datasets'][f'Fold-{fold}']['Total-Features'], sep=',', index=False)
    else:
        print(f"[Fold-{fold}]\tAll Features have already been Extracted!")

def extractImportant1DFeatures(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract all the important 1D features 
        from the audio samples of the selected Fold on the dataset
        as well as normalizing them and leaving them ready to be used.
    ------------------------------------------------------------------
    := param: audio_df - Pandas Dataframe with the UrbamSound8K dataset metadata.
    := param: fold - Fold of the audios that we want to perform feature extraction on.
    := param: config - Dictionary with Constants used in audio processing throughout the project.
    := param: pathsConfig - Dictionary with filepaths to help organize the results of the project.
    := return: None, since we are merely extracting data.
    """

    # Check if the dataframe has already been computed
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['Total-Features']):
        # Initialize a List to store the extracted content
        data = []

        # Get the audio filenames from the selected fold
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Iterate through all the audios inside the selected fold
        for audioFileName in foldAudios:
            # Load the Audio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # Compute features

            # [1D Features]
            # Zero Crossing Rate
            zeroCrossingRate = librosa.feature.zero_crossing_rate(y=audio).tolist()[0]

            # Spectral Centroid
            spectralCentroid = librosa.feature.spectral_centroid(y=audio, sr=config['SAMPLE_RATE']).tolist()[0]

            # Spectral Bandwidth
            spectralBandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=config['SAMPLE_RATE']).tolist()[0]

            # Spectral Flatness
            spectralFlatness = librosa.feature.spectral_flatness(y=audio).tolist()[0]

            # Spectral Roll-off
            spectralRolloff = librosa.feature.spectral_rolloff(y=audio, sr=config['SAMPLE_RATE']).tolist()[0]

            # RMS Energy
            rmsEnergy = librosa.feature.rms(y=audio).tolist()[0]

            # [2D Features]

            # MFCCs
            mfccRaw = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])
            mfcc = np.mean(mfccRaw, axis=1).tolist()

            # Chroma STFT
            chromaSTFTRaw = librosa.feature.chroma_stft(y=audio, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])
            chromaSTFT = np.mean(chromaSTFTRaw, axis=1).tolist()

            # Mel Spectrogram
            melSpectrogramRaw = librosa.feature.melspectrogram(y=audio, sr=config['SAMPLE_RATE'])
            melSpectrogram = np.mean(melSpectrogramRaw, axis=1).tolist()

            # Spectral Contrast
            spectralContrastRaw = librosa.feature.spectral_contrast(y=audio, sr=config['SAMPLE_RATE'])
            spectralContrast = np.mean(spectralContrastRaw, axis=1).tolist()

            # Append the extracted features to the data list
            data.append({
                # Audio Details
                'audio':audioFileName,
                'fold':fold,

                # 1 Dimensional Features
                'Zero-Crossing Rate': zeroCrossingRate,
                'Spectral Centroid': spectralCentroid,
                'Spectral Bandwidth':spectralBandwidth,
                'Spectral Flatness':spectralFlatness,
                'Spectral Roll-off':spectralRolloff,
                'RMS Energy':rmsEnergy,

                # 2 Dimensional Features
                'MFCC':mfcc,
                'Chroma STFT':chromaSTFT,
                'Mel Spectrogram':melSpectrogram,
                'Spectral Contrast':spectralContrast,

                # Target
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

        # Create a DataFrame with the collected data
        df = pd.DataFrame(data)

        # Save the Dataframe
        df.to_csv(pathsConfig['Datasets'][f'Fold-{fold}']['Total-Features'], sep=',', index=False)
    else:
        print(f"[Fold-{fold}]\tImportant Features have already been Extracted!")