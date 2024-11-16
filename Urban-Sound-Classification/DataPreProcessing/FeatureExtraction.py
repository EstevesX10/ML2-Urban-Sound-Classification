import numpy as np
import pandas as pd
import os
import librosa
from .AudioManagement import (loadAudio)
        
def extractAllRawFeatures(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
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
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['All-Raw-Features']):
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
            zeroCrossingRate = librosa.feature.zero_crossing_rate(y=audio)[0]

            # Spectral Centroid
            spectralCentroid = librosa.feature.spectral_centroid(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Spectral Bandwidth
            spectralBandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Spectral Flatness
            spectralFlatness = librosa.feature.spectral_flatness(y=audio)[0]

            # Spectral Roll-off
            spectralRolloff = librosa.feature.spectral_rolloff(y=audio, sr=config['SAMPLE_RATE'])[0]

            # RMS Energy
            rmsEnergy = librosa.feature.rms(y=audio)[0]

            # [2-Dimensional Features]
            # MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])
            
            # Chroma STFT
            chromaSTFT = librosa.feature.chroma_stft(y=audio, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])

            # Mel Spectrogram
            melSpectrogram = librosa.feature.melspectrogram(y=audio, sr=config['SAMPLE_RATE'])

            # Spectral Contrast
            spectralContrast = librosa.feature.spectral_contrast(y=audio, sr=config['SAMPLE_RATE'])

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
        df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}']['All-Raw-Features'])

def extractRawFeatures1D(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract all the raw 1D features 
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
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['1D-Raw-Features']):
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
            zeroCrossingRate = librosa.feature.zero_crossing_rate(y=audio)[0]

            # Spectral Centroid
            spectralCentroid = librosa.feature.spectral_centroid(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Spectral Bandwidth
            spectralBandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=config['SAMPLE_RATE'])[0]

            # Spectral Flatness
            spectralFlatness = librosa.feature.spectral_flatness(y=audio)[0]

            # Spectral Roll-off
            spectralRolloff = librosa.feature.spectral_rolloff(y=audio, sr=config['SAMPLE_RATE'])[0]

            # RMS Energy
            rmsEnergy = librosa.feature.rms(y=audio)[0]

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

                # Target
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

        # Create a DataFrame with the collected data
        df = pd.DataFrame(data)

        # Save the Dataframe
        df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}']['1D-Raw-Features'])

def extractRawFeatures2D(audio_df:pd.DataFrame, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract all the raw 2D features 
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
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}']['2D-Raw-Features']):
        # Initialize a List to store the extracted content
        data = []

        # Get the audio filenames from the selected fold
        foldAudios = audio_df[audio_df['fold'] == fold]['slice_file_name'].to_numpy()

        # Iterate through all the audios inside the selected fold
        for audioFileName in foldAudios:
            # Load the Audio
            audio = loadAudio(df_audio=audio_df, audioSliceName=audioFileName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)
    
            # [2-Dimensional Features]
            # MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])
            
            # Chroma STFT
            chromaSTFT = librosa.feature.chroma_stft(y=audio, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])

            # Mel Spectrogram
            melSpectrogram = librosa.feature.melspectrogram(y=audio, sr=config['SAMPLE_RATE'])

            # Spectral Contrast
            spectralContrast = librosa.feature.spectral_contrast(y=audio, sr=config['SAMPLE_RATE'])

            # Compute and append the extracted features to the data list
            data.append({
                # Audio Details
                'audio':audioFileName,
                'fold':fold,
                
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
        df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}']['2D-Raw-Features'])
        
def getFeaturesDetails(df:pd.DataFrame, intervalStep:int) -> list[dict]:
        """
        # Description
            -> Based on the extracted features from the first audio sample, it computes the amount of
            components we need to use to partition each feature while considering the possible residue.
        -----------------------------------------------------------------------------------------------
        := param: df - Pandas DataFrame in which we want to extract the features's details.
        := param: stepInterval - Window Size of the segmentation we want to consider when creating components for the extracted features.
        := return: A list with all the data regarding the data formating on the next step.
        """

        # Selecting the important columns to extract the details from
        cols = df.columns[2:len(df.columns) - 2]

        # List to store the details of the columns to process
        columnsDetails = []

        # Iterate over the features of the DataFrame
        for feature in cols:
            # Analyse the shape of the array of the current feature
            length = len(df.iloc[0][feature])

            # Calculate the number of components for the current features
            numComponents = length // intervalStep
            residueSize = length / intervalStep

            # Update the initial list with the computed data
            columnsDetails.append({
                'feature':feature.replace('-', '_').replace(' ', '_'),
                'totalComponents':numComponents,
                'residueSize':residueSize
            })
        
        return columnsDetails

def processRawFeatures(fold:int, intervalStep:int, featuresDimensionality:str, pathsConfig:dict) -> None:
    """
    # Description
        -> This Method allows to process the previously extracted raw features into
        multiple components based on small partitions of the data and a couple metrics.
    -------------------------------------------------------------------------------------
    := param: fold - Fold in which we want to process the extracted features.
    := param: intervalStep - Window Size of the segmentation we want to consider when creating components for the extracted features.
    := param: featureDimensionality - Dimensionality of the data to process (either "1D" or "2D").
    := return: None, since we are only updating one attribute of the class.
    """

    # Verify the integrity of the feature dimensionality chosen
    if featuresDimensionality not in ["1D", "2D"]:
        raise ValueError("Invalid Feature Dimensionality Chosen")

    # Create a variable for the processed dataframe
    processed_df = None

    # If the DataFrame with the processed data has not been computed
    if not os.path.exists(pathsConfig['Datasets'][f'Fold-{fold}'][f'{featuresDimensionality}-Processed-Features']):

        # Load the dataset with the raw features and select the important columnhs
        df = pd.read_pickle(pathsConfig['Datasets'][f'Fold-{fold}'][f'{featuresDimensionality}-Raw-Features'])
        featuresToProcess = df.columns[2:len(df.columns) - 2]

        # Fetch the Column's Details
        columnDetails = getFeaturesDetails(df, intervalStep)

        # Iterate row by row and process each extracted vector to get mean, std, ... to obtain multiple columns [MEAN_F1, STD_F1, MEAN_F2, STD_F2, ...]
        for index, row in df.iterrows():
            # Create a new dictionary for a new line in the Dataframe
            audioSampleData = {'audio':row['audio'], 'fold':fold}

            # Create a featureIdx to keep track of the current feature being analysed
            featureIdx = 0

            if featuresDimensionality == "1D":
                # Iterate through the 1-Dimensional Features
                for feature in featuresToProcess:
                    # Fetch the array in the current cell
                    featureArray = row[feature]

                    # Create the components for the 1-Dimensional Data
                    for currentComponent in range(1, columnDetails[featureIdx]['totalComponents'] + 1):
                        if currentComponent == columnDetails[featureIdx]['totalComponents'] - 1: 
                            audioSampleData.update({
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Mean":np.mean(featureArray[currentComponent*intervalStep :]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Median":np.median(featureArray[currentComponent*intervalStep :]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Std":np.std(featureArray[currentComponent*intervalStep :])
                            })
                        else:
                            audioSampleData.update({
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Mean":np.mean(featureArray[(currentComponent - 1)*intervalStep : currentComponent*intervalStep]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Median":np.median(featureArray[(currentComponent - 1)*intervalStep : currentComponent*intervalStep]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Std":np.std(featureArray[(currentComponent - 1)*intervalStep : currentComponent*intervalStep])
                            })
                    
                    # Increment the index of the feature being processed
                    featureIdx += 1
            
            elif featuresDimensionality == "2D":
                # Iterate through the 2-Dimensional Features
                for feature in featuresToProcess:
                    # Fetch and Convert the array in the current cell
                    featureArray = np.mean(row[feature], axis=1)

                    # Update the audio Sample Data with all the components previously calculated during feature extraction
                    for componentIdx, component in enumerate(featureArray):
                        audioSampleData.update({
                            f"{columnDetails[featureIdx]['feature']}_{componentIdx}":component
                        })
                    
                    # Increment the index of the feature being processed
                    featureIdx += 1

            # Add the target Label
            audioSampleData.update({
                'target':row['target']
            })

            # Check if we already have a DataFrame
            if processed_df is None:
                # Create a new one from zero
                processed_df = pd.DataFrame([audioSampleData])
            else:
                # Create a new DataFram with the new processed audio entry
                newLine = pd.DataFrame([audioSampleData])

                # Concatenate the new DataFrame with the previous one
                processed_df = pd.concat([processed_df, newLine], ignore_index=True)
        
        # Save the Processed data
        processed_df.to_pickle(pathsConfig['Datasets'][f'Fold-{fold}'][f'{featuresDimensionality}-Processed-Features'])

def extractMFCCs(audio_df:pd.DataFrame, raw:bool, fold:int, config:dict, pathsConfig:dict) -> None:
    """
    # Description
        -> This function helps extract the MFCCs from audio samples 
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
        mfccsFilePath = pathsConfig['Datasets'][f'Fold-{fold}']['2D-Raw-MFCCs']
    else:
        mfccsFilePath = pathsConfig['Datasets'][f'Fold-{fold}']['1D-Processed-MFCCs']

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
    
            # Define a dictionary for the current audio data
            audioData = {
                # Audio Details
                'audio':audioFileName,
                'fold':fold,
            }

            # Compute the mfccs
            if raw: 
                # Raw Data
                mfcc = librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC'])

                # Update the previous audio Data
                audioData.update({
                    # MFCC
                    'MFCC':mfcc,
                })

            else: 
                # Mean on each coefficient
                mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=config['SAMPLE_RATE'], n_mfcc=config['N_MFCC']), axis=1)

                # Update the audio Sample Data with all the components of the MFCCs (each coeficient's mean)
                for componentIdx, component in enumerate(mfcc):
                    audioData.update({
                        f"MFCC_{componentIdx}":component
                    })

            # Add the target
            audioData.update({
                # Target
                'target':audio_df[audio_df['slice_file_name'] == audioFileName]['class'].to_numpy()[0]
            })

            # Append the audio data to the previous audio's list
            data.append(audioData)

        # Create a DataFrame with the collected data
        df = pd.DataFrame(data)

        # Save the Dataframe
        df.to_pickle(mfccsFilePath)
