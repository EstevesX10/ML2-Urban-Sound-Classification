import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.callbacks.history import History
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, RobustScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scikit_posthocs as sp


def plotNetworkTrainingPerformance(
    confusionMatrix: np.ndarray, title: str, trainHistory: History, targetLabels=None
) -> None:
    """
    # Description
        -> This function helps visualize the network's performance
        during training through it's variation on both loss and accuracy.
    ---------------------------------------------------------------------
    := param: confusionMatrix - Confusion Matrix obtained from the given model.
    := param: trainHistory - Network's training history data.
    := param: targetLabels - Target Labels of the UrbanSound8k dataset.
    := return: None, since we are simply plotting data.
    """

    # Create a figure with axis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Set the overall title for the entire figure
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Plot training & validation accuracy values
    ax1.plot(trainHistory["accuracy"], label="Train Accuracy")
    ax1.plot(trainHistory["val_accuracy"], label="Validation Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(loc="lower right")

    # Plot training & validation loss values
    ax2.plot(trainHistory["loss"], label="Train Loss")
    ax2.plot(trainHistory["val_loss"], label="Validation Loss")
    ax2.set_title("Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="upper right")

    # Plot Confusion Matrix
    plotConfusionMatrix(confusionMatrix, targetLabels=targetLabels, ax=ax3)

    plt.tight_layout()
    plt.show()


def plotConfusionMatrix(
    confusionMatrix, title="Confusion Matrix", targetLabels=None, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()

    # Plot Confusion matrix
    sns.heatmap(
        confusionMatrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=targetLabels,
        yticklabels=targetLabels,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")


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


def plotScatterClass(
    X: np.array,
    targets: np.array,
    xLogScale: bool = False,
    algorithm: str = "PCA",
    randomState=42,
):
    # Encode labels
    encoder = LabelEncoder()
    targetsEncoded = encoder.fit_transform(targets)

    # Apply dimensionality reduction
    if algorithm == "PCA":
        X_embedded = PCA(n_components=2, random_state=randomState).fit_transform(X)
    elif algorithm == "t-sne":
        if X.shape[1] > 10:
            # Reduce the number of dimensions to a reasonable amount
            X = PCA(n_components=10, random_state=randomState).fit_transform(X)

        X_embedded = TSNE(
            n_components=2,
            learning_rate="auto",
            init="pca",
            perplexity=30,
            random_state=randomState,
            n_jobs=-1,
        ).fit_transform(X)
    else:
        raise ValueError(f"Invalid algorithm {algorithm}")

    X_embedded = RobustScaler().fit_transform(X_embedded)

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap("Paired", len(encoder.classes_))
    plt.scatter(
        X_embedded[:, 0], X_embedded[:, 1], c=targetsEncoded[:], cmap=colors, s=10
    )

    # Add labels, title, and legend
    plt.xlabel(f"{algorithm} Dimension 1")
    plt.ylabel(f"{algorithm} Dimension 1")
    plt.title(f"{algorithm} Plot of High-Dimensional Data")

    if xLogScale:
        plt.xscale("log")

    # Add a legend to show class mapping
    legend_labels = [
        mpatches.Patch(color=colors(i), label=label)
        for i, label in enumerate(encoder.classes_)
    ]
    plt.legend(handles=legend_labels, title="Class")
