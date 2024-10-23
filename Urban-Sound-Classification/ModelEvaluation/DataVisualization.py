import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp

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