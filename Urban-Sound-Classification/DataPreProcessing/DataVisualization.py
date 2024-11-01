import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import librosa
from .AudioManagement import (loadAudio)
from PIL import Image
import matplotlib.image as mpimg

def pastelizeColor(c:tuple, weight:float=None) -> np.ndarray:
    """
    # Description
        -> Lightens the input color by mixing it with white, producing a pastel effect.
    -----------------------------------------------------------------------------------
    := param: c - Original color.
    := param: weight - Amount of white to mix (0 = full color, 1 = full white).
    """

    # Set a default weight
    weight = 0.5 if weight is None else weight

    # Initialize a array with the white color values to help create the pastel version of the given color
    white = np.array([1, 1, 1])

    # Returns a tuple with the values for the pastel version of the color provided
    return mcolors.to_rgba((np.array(mcolors.to_rgb(c)) * (1 - weight) + white * weight))

def plotFeatureDistribution(df:pd.DataFrame=None, classFeature:str=None, forceCategorical:bool=None, pathsConfig:dict=None, featureDecoder:dict=None) -> None:
    """
    # Description
        -> This function plots the distribution of a feature (column) in a dataset.
    -------------------------------------------------------------------------------
    := param: df - Pandas DataFrame containing the dataset metadata.
    := param: feature - Feature of the dataset to plot.
    := param: forceCategorical - Forces a categorical analysis on a numerical feature.
    := param: pathsConfig - Dictionary with important paths used to store some plots.
    := param: featureDecoder - Dictionary with the conversion between the column value and its label [From Integer to String].
    """

    # Check if a dataframe was provided
    if df is None:
        print('The dataframe was not provided.')
        return
    
    # Check if a feature was given
    if classFeature is None:
        print('Missing a feature to Analyse.')
        return

    # Check if the feature exists on the dataset
    if classFeature not in df.columns:
        print(f"The feature '{classFeature}' is not present in the dataset.")
        return

    # Set default value
    forceCategorical = False if forceCategorical is None else forceCategorical

    # Define a file path to store the final plot
    if pathsConfig is not None:
        savePlotPath = pathsConfig['ExploratoryDataAnalysis'] + '/' + f'{classFeature}Distribution.png'
    else:
        savePlotPath = None

    # Define a Figure size
    figureSize = (8,5)

    # Check if the plot has already been computed
    if savePlotPath is not None and os.path.exists(savePlotPath):
        # Load the image file with the plot
        plot = mpimg.imread(savePlotPath)

        # Get the dimensions of the plot in pixels
        height, width, _ = plot.shape

        # Set a DPI value used to previously save the plot
        dpi = 100

        # Create a figure with the exact same dimensions as the previouly computed plot
        _ = plt.figure(figsize=(width / 2 / dpi, height / 2 / dpi), dpi=dpi)

        # Display the plot
        plt.imshow(plot)
        plt.axis('off')
        plt.show()
    else:
        # Check the feature type
        if pd.api.types.is_numeric_dtype(df[classFeature]):
            # For numerical class-like features, we can treat them as categories
            if forceCategorical:
                # Create a figure
                _ = plt.figure(figsize=figureSize)

                # Get unique values and their counts
                valueCounts = df[classFeature].value_counts().sort_index()
                
                # Check if a feature Decoder was given and map the values if possible
                if featureDecoder is not None:
                    # Map the integer values to string labels
                    labels = valueCounts.index.map(lambda x: featureDecoder.get(x, x))
                    
                    # Tilt x-axis labels by 0 degrees and adjust the fontsize
                    plt.xticks(rotation=0, ha='center', fontsize=8)
                
                # Use numerical values as the class labels
                else:
                    labels = valueCounts.index

                # Create a color map from green to red
                cmap = plt.get_cmap('RdYlGn_r')  # Reversed 'Red-Yellow-Green' colormap (green to red)
                colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plot the bars with gradient colors
                bars = plt.bar(labels.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plot the grid behind the bars
                plt.grid(True, zorder=1)

                # Add text (value counts) to each bar at the center with a background color
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Use a lighter color as the background for the text
                    lighterColor = pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Add title and labels
                plt.title(f'Distribution of {classFeature}')
                plt.xlabel(f'{classFeature} Labels', labelpad=20)
                plt.ylabel('Number of Samples')
                
                # Save the plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Display the plot
                plt.show()
            
            # For numerical features, use a histogram
            else:
                # Create a figure
                plt.figure(figsize=figureSize)

                # Plot the histogram with gradient colors
                plt.hist(df[classFeature], bins=30, color='lightgreen', edgecolor='lightgrey', alpha=1.0, zorder=2)
                
                # Add title and labels
                plt.title(f'Distribution of {classFeature}')
                plt.xlabel(classFeature)
                plt.ylabel('Frequency')
                
                # Tilt x-axis labels by 0 degrees and adjust the fontsize
                plt.xticks(rotation=0, ha='center', fontsize=10)

                # Plot the grid behind the bars
                plt.grid(True, zorder=1)
                
                # Save the plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Display the plot
                plt.show()

        # For categorical features, use a bar plot
        elif pd.api.types.is_categorical_dtype(df[classFeature]) or df[classFeature].dtype == object:
                # Create a figure
                plt.figure(figsize=figureSize)

                # Get unique values and their counts
                valueCounts = df[classFeature].value_counts().sort_index()
                
                # Create a color map from green to red
                cmap = plt.get_cmap('viridis')  # Reversed 'Red-Yellow-Green' colormap (green to red)
                colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]

                # Plot the bars with gradient colors
                bars = plt.bar(valueCounts.index.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
                
                # Plot the grid behind the bars
                plt.grid(True, zorder=1)

                # Add text (value counts) to each bar at the center with a background color
                for i, bar in enumerate(bars):
                    yval = bar.get_height()
                    # Use a lighter color as the background for the text
                    lighterColor = pastelizeColor(colors[i], weight=0.2)
                    plt.text(bar.get_x() + bar.get_width() / 2,
                            yval / 2,
                            int(yval),
                            ha='center',
                            va='center',
                            fontsize=10,
                            color='black',
                            bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.3'))

                # Add title and labels
                plt.title(f'Distribution of {classFeature}')
                plt.xlabel(f'{classFeature} Labels', labelpad=20)
                plt.ylabel('Number of Samples')
                
                # Tilt x-axis labels by 0 degrees and adjust the fontsize
                plt.xticks(rotation=25, ha='center', fontsize=8)

                # Save the plot
                if savePlotPath is not None and not os.path.exists(savePlotPath):
                    plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

                # Display the plot
                plt.show()
        
        # Unknown Behaviour
        else:
            print(f"The feature '{classFeature}' is not supported for plotting.")

def plotFeatureDistributionByFold(df:pd.DataFrame=None, classFeature:str=None, foldFeature:str=None, pathsConfig:dict=None, featureDecoder:dict=None) -> None:
    """
    # Description
        -> Plots the class distribution for each fold in the dataset.
    -----------------------------------------------------------------
    := param: df - Pandas DataFrame containing the dataset metadata.
    := param: classFeature - The class feature of the dataset.
    := param: foldFeature - The feature that indicates the fold.
    := param: pathsConfig - Dictionary with important paths used to store some plots.
    := param: featureDecoder - Dictionary with the conversion between the class value and its label.
    """

    # Check if DataFrame, class feature, and fold feature were provided
    if df is None or classFeature is None or foldFeature is None:
        print('DataFrame, class feature, or fold feature is missing.')
        return
    
    # Check if the features exist in the DataFrame
    if classFeature not in df.columns or foldFeature not in df.columns:
        print(f"Either '{classFeature}' or '{foldFeature}' is not present in the dataset.")
        return
    
    # Define a path to save the plot
    if pathsConfig is not None:
        savePlotPath = pathsConfig['ExploratoryDataAnalysis'] + '/' + f'{classFeature}DistributionPerFold.png'
    else:
        savePlotPath = None

    # Define a Figure size
    figureSize = (18, 8)

    # Check if the plot has already been computed
    if savePlotPath is not None and os.path.exists(savePlotPath):
        # Load the image file with the plot
        plot = mpimg.imread(savePlotPath)

        # Get the dimensions of the plot in pixels
        height, width, _ = plot.shape

        # Set a DPI value used to previously save the plot
        dpi = 300  

        # Create a figure with the exact same dimensions as the previouly computed plot
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

        # Display the plot
        plt.imshow(plot)
        plt.axis('off')
        plt.show()

    else:
        # Get the unique folds
        uniqueFolds = sorted(df[foldFeature].unique())
        
        # Set up a 2x5 grid for the subplots
        fig, axes = plt.subplots(2, 5, figsize=figureSize)
        fig.suptitle(f'{classFeature} Distribution on Each Fold', fontsize=16, y=0.95)
        
        for i, fold in enumerate(uniqueFolds):
            # Filter the DataFrame for the current fold
            foldData = df[df[foldFeature] == fold]
            
            # Get value counts for the class feature
            valueCounts = foldData[classFeature].value_counts().sort_index()
            
            # Use the featureDecoder if provided
            if featureDecoder is not None:
                labels = valueCounts.index.map(lambda x: featureDecoder.get(x, x))
            else:
                labels = valueCounts.index
            
            # Create a color map for the bars
            cmap = plt.get_cmap('viridis')
            colors = [pastelizeColor(cmap(i / (len(valueCounts) - 1))) for i in range(len(valueCounts))]
            
            # Get the row and column index for the subplot grid
            row, col = divmod(i, 5)
            
            # Plot the bars in the correct subplot
            # bars = axes[row, col].bar(labels.astype(str), valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.4, zorder=2)

            # Adjust the positions of the bars and increase their thickness
            positions = np.arange(len(labels))  # Positions for the bars
            bars = axes[row, col].bar(positions, valueCounts.values, color=colors, edgecolor='lightgrey', alpha=1.0, width=0.8, zorder=2)
            
            # Add text (value counts) to each bar at the center
            for j, bar in enumerate(bars):
                yval = bar.get_height()
                lighterColor = pastelizeColor(colors[j], weight=0.2)
                axes[row, col].text(bar.get_x() + bar.get_width() / 2,
                                    yval / 2,
                                    int(yval),
                                    ha='center',
                                    va='center',
                                    fontsize=7,
                                    color='black',
                                    bbox=dict(facecolor=lighterColor, edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Add title and labels
            axes[row, col].set_title(f'[Fold] {fold}', fontsize=12)
            axes[row, col].set_xlabel(f'{classFeature} Labels', fontsize=10)
            axes[row, col].set_ylabel('Number of Samples', fontsize=10)
            axes[row, col].grid(True, zorder=1)
            axes[row, col].set_xticks(positions)
            axes[row, col].set_xticklabels(labels.astype(str), rotation=50, ha='right', fontsize=8)
            
        # Adjust layout for better display
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save the plot to a file
        if savePlotPath is not None and not os.path.exists(savePlotPath):
            plt.savefig(savePlotPath, dpi=300, bbox_inches='tight')

        plt.show()

def plotAudioWave(df_audio:pd.DataFrame=None, audioSliceName:str=None, config:dict=None) -> None:
    """
    # Description
        -> This function plots the selected audio's wave form. 
    ----------------------------------------------------------
    := param: df_audio - Pandas DataFrame with the dataset's metadata.
    := param: audioSliceName - Audio Identification inside the dataset.
    := param: config - Dictionary with important values used during audio processing tasks within the project.
    := return: None, since we are only plotting the audio's wave.
    """
    
    # Check if the dataframe with the dataset's metadata was given
    if df_audio is None:
        raise ValueError("Missing a DataFrame with the dataset's metadata!")

    # Check if a audio was selected
    if audioSliceName is None:
        raise ValueError("Missing a Audio to plot the wave form of!")
    
    # Check if the config dictionary was provided
    if config is None:
        raise ValueError("Missing the Configuration dictionary!")

    # Load the Audio
    data = loadAudio(df_audio=df_audio, audioSliceName=audioSliceName, audioDuration=config['DURATION'], targetSampleRate=config['SAMPLE_RATE'], usePadding=True)

    # Set up the plot
    plt.figure(figsize=(12,4))

    # Plot the waveform
    librosa.display.waveshow(data, sr=config['SAMPLE_RATE'], color='dodgerblue', alpha=0.7)
    plt.fill_between(config['DURATION'], data, color='dodgerblue', alpha=0.3)  # Fill color for better readability

    # Add title and labels
    plt.title('[Normalized] Audio Wave', fontsize=16, pad=20)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Amplitude", fontsize=14)

    # Customize grid and spines
    plt.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plotAudio1DimensionalFeature(audioData:np.ndarray=None, extracted1DimensionalFeature:np.ndarray=None, featureName:str=None, yLabel:str=None, color:str=None, config:dict=None) -> None:
    """
    # Description
        -> This function helps plot all the extracted 1 Dimensional Features.
    -------------------------------------------------------------------------
    := param: audioData - Audio Time Series within a numpy array.
    := param: extracted1DimensionalFeature - Numpy Array with the 1 Dimensional Feature extracted from the AudioData.
    := param: featureName - Name of the Feature extracted from the audio.
    := param: yLabel - Label for the y-axis.
    := param: color - Color for the line on the feature's plot.
    := param: config - Dictionary with important values used during audio processing tasks within the project.
    := return: None, since we are only displaying the features.
    """
   
    # Check if a audio time series was provided
    if audioData is None:
        raise ValueError("Missing a Audio Time Series!")

    # Check if a extracted feature was provided
    if extracted1DimensionalFeature is None:
        raise ValueError("Missing extracted Feature")
    
    # Check extracted feature shape
    if extracted1DimensionalFeature.shape[0] > 1:
        raise ValueError("Shape mismatch of the extracted feature [It does not correspond to a 1-Dimensional Feature]")

    # Check if a feature name was given
    if featureName is None:
        raise ValueError("Missing a Feature Name!")

    # Verify if a y Label description was given
    if yLabel is None:
        raise ValueError("Missing a Y-Axis Description!")

    # Check if a config was given
    if config is None:
        raise ValueError("Missing a configuration dictionary!")

    # Setting a default color    
    color = 'r' if color is None else color

    # Flatten the extracted 1-D feature
    flattenedFeature = extracted1DimensionalFeature.flatten()
    
    # Apply a style sheet to the plot
    plt.style.use('seaborn-v0_8-pastel')

    # Create a figure
    plt.figure(figsize=(10, 6))
    
    # Plot the audio waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audioData, sr=config['SAMPLE_RATE'], alpha=0.6, color='dodgerblue')
    plt.title('[Normalized] Audio Wave', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True)
    
    # Plot the flattened feature
    plt.subplot(2, 1, 2)
    frames = range(len(flattenedFeature))
    t = librosa.frames_to_time(frames, sr=config['SAMPLE_RATE'])
    plt.plot(t, flattenedFeature, color=color, label=featureName)
    plt.title(f'{featureName}', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel(yLabel, fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=12)
    
    # Adjust the layout
    plt.tight_layout(pad=3.0)
    plt.show()

def plotChromaFeatures(audioData:np.ndarray=None, config:dict=None) -> None:
    """
    # Description
        -> This function plots the audio's chroma features.
    -------------------------------------------------------
    := param: audioData - Audio Time Series within a numpy array.
    := param: config - Dictionary with important values used during audio processing tasks within the project.
    := return: None, since we are simply plotting some data.
    """
    
    # Check if a audio time series was provided
    if audioData is None:
        raise ValueError("Missing a Audio Time Series!")

    # Check if a config was given
    if config is None:
        raise ValueError("Missing a configuration dictionary!")

    # Compute the Chroma Features
    chroma_stft = librosa.feature.chroma_stft(y=audioData, n_chroma=config['N_CHROMA'], sr=config['SAMPLE_RATE'], n_fft=config['N_FFT'], hop_length=config['HOP_LENGTH'], win_length=config['WINDOW_LENGTH'])
    
    # Create a figure
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', sr=config['SAMPLE_RATE'], cmap='PuBu')
    plt.colorbar()

    # Create a legend for the axis and add a title
    plt.title('Chroma Features (STFT)', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Chroma', fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plotMelSpectrogram(audioData:np.ndarray=None, sample_rate:int=None, n_mels:int=None, fmin:int=None, fmax=None) -> None:
    """
    # Description
        -> Plots a Mel-frequency spectrogram.
    -----------------------------------------
    Parameters:
    := param: audio - Audio time series.
    := param: sample_rate - Sample rate of the audio.
    := param: n_mels - Number of Mel bands to generate.
    := param: fmin - Minimum frequency for the Mel filter bank.
    := param: fmax - Maximum frequency for the Mel filter bank.
    := return: None, since we are only plotting the mel spectrogram.
    """

    # Check if a audio time series was provided
    if audioData is None:
        raise ValueError("Missing a Audio Time Series!")
    
    # Check if a Sample Rate was givem
    if sample_rate is None:
        raise ValueError("Missing a Sample Rate!")
    
    # Setting default Values
    n_mels = 128 if n_mels is None else n_mels
    fmin = 0 if fmin is None else fmin

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audioData, sr=sample_rate, n_mels=n_mels, fmin=fmin, fmax=fmax)

    # Convert to dB for visualization
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Create a figure for the spectrogram
    plt.figure(figsize=(9, 5))
    ax = plt.axes()
    spec = librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax, ax=ax, cmap='PuBu')

    # Customize colorbar
    cbar = plt.colorbar(spec, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)', rotation=270, labelpad=15, fontsize=12)

    # Add both labels and title
    plt.title('Mel-frequency Spectrogram', fontsize=16, pad=20)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Frequency (Mel)", fontsize=14)
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)

    plt.tight_layout(pad=2.0)
    plt.show()

def plotSpectralContrast(audioData:np.ndarray=None, config:dict=None) -> None:
    """
    # Description
        -> This function plots the audio's spectral contrast.
    ---------------------------------------------------------
    := param: audioData - Audio Time Series within a numpy array.
    := param: config - Dictionary with important values used during audio processing tasks within the project.
    := return: None, since we are simply plotting some data.
    """
    
    # Check if a audio time series was provided
    if audioData is None:
        raise ValueError("Missing a Audio Time Series!")

    # Check if a config was given
    if config is None:
        raise ValueError("Missing a configuration dictionary!")

    # Compute the Spectral Contrast
    spectralContrast = librosa.feature.spectral_contrast(y=audioData, sr=config['SAMPLE_RATE'])
    
    # Create a figure
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectralContrast, x_axis='time', sr=config['SAMPLE_RATE'], cmap='PuBu')
    plt.colorbar(label='Spectral Contrast (dB)')

    # Create a legend for the axis and add a title
    plt.title('Spectral Contrast', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency Bands', fontsize=14)

    # Show the plot
    plt.tight_layout()
    plt.show()