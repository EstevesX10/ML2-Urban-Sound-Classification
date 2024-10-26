import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
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
    := param: df - Pandas DataFrame containing the dataset.
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
    := param: df - Pandas DataFrame containing the dataset.
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