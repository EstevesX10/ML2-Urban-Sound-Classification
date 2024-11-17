import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.src.callbacks.history import History
from sklearn.metrics import confusion_matrix, classification_report
import scikit_posthocs as sp
from sklearn.metrics import ConfusionMatrixDisplay

def plotNetworkTrainingPerformance(model:keras.Model, X_test:np.ndarray, y_test:np.ndarray, trainHistory:History=None, targetLabels=None) -> None:
    """
    # Description
        -> This function helps visualize the network's performance
        during training through it's variation on both loss and accuracy.
    ---------------------------------------------------------------------
    := param: model - Keras model instance.
    := param: X_test - Test Set [Features].
    := param: y_test - Test Set [Target Label].
    := param: trainHistory - Network's training history data.
    := param: targetLabels - Target Labels of the UrbanSound8k dataset.
    := return: None, since we are simply plotting data.
    """

    # Create a figure with axis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

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

    # Confusion matrix
    if model is not None and X_test is not None and y_test is not None:
        # Get predictions
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=targetLabels, yticklabels=targetLabels)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted Labels')
        ax3.set_ylabel('True Labels')
    else:
        ax3.axis('off')  # Hide the confusion matrix plot if data is not provided
        print("Confusion Matrix not plotted because model, X_test, or y_test is missing.")

    plt.tight_layout()
    plt.show()


def plotCritialDifferenceDiagram(
    matrix: np.ndarray = None, colors: dict = None
) -> None:
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
        raise ValueError(
            "Failed to get a dictionary with the colors for the Critical Difference Diagram"
        )

    # Calculate ranks
    ranks = matrix.rank(axis=1, ascending=False).mean()

    # Perform Nemenyi post-hoc test
    nemenyi = sp.posthoc_nemenyi_friedman(matrix)

    # Add Some Styling
    marker = {"marker": "o", "linewidth": 1}
    labelProps = {"backgroundcolor": "#ADD5F7", "verticalalignment": "top"}

    # Plot the Critical Difference Diagram
    _ = sp.critical_difference_diagram(
        ranks,
        nemenyi,
        color_palette=colors,
        marker_props=marker,
        label_props=labelProps,
    )
