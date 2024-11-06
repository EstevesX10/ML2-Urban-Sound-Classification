import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.callbacks.history import History
import scikit_posthocs as sp

def plotNetworkTrainingPerformance(trainHistory:History=None) -> None:
    """
    # Description
        -> This function helps visualize the network's performance 
        during training through it's variation on both loss and accuracy.
    ---------------------------------------------------------------------
    := param: trainHistory - Network's training history data.
    := return: None, since we are simply plotting data.
    """

    # Check if a Network train history was passed on
    if trainHistory is None:
        raise ValueError("Missing the Training History Data of the Network!")

    # Create a figure with axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot training & validation accuracy values
    ax1.plot(trainHistory.history['accuracy'], label='Train Accuracy')
    ax1.plot(trainHistory.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')

    # Plot training & validation loss values
    ax2.plot(trainHistory.history['loss'], label='Train Loss')
    ax2.plot(trainHistory.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plotCritialDifferenceDiagram(matrix:np.ndarray=None, colors:dict=None) -> None:
    """
    # Description
        -> Plots the Critical Difference Diagram.
    ---------------------------------------------
    := param: matrix - Dataframe with the Accuracies obtained by the Models.
    := param: colors - Dictionary that matches each column of the df to a color to use in the Diagram.
    := return: None, since we are simply ploting a diagram. 
    """
    
    # Check if the matrix was passed
    if matrix is None:
        raise ValueError("Missing a Matrix!")
    
    # Check if a colors dictionary was provides
    if colors is None:
        raise ValueError("Failed to get a dictionary with the colors for the Critical Difference Diagram")

    # Calculate ranks
    ranks = matrix.rank(axis=1, ascending=False).mean()
    
    # Perform Nemenyi post-hoc test
    nemenyi = sp.posthoc_nemenyi_friedman(matrix)

    # Add Some Styling
    marker = {'marker':'o', 'linewidth':1}
    labelProps = {'backgroundcolor':'#ADD5F7', 'verticalalignment':'top'}
    
    # Plot the Critical Difference Diagram
    _ = sp.critical_difference_diagram(ranks, nemenyi, color_palette=colors, marker_props=marker, label_props=labelProps)