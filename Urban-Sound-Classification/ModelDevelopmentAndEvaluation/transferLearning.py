import pandas as pd
import librosa as libr
from Utils.Configuration import loadConfig
from DataPreProcessing.AudioManagement import formatFilePath
import numpy as np
from typing import List

import tensorflow as tf
import tensorflow_hub as hub

YAMNET_SAMPLE_RATE = 16_000


def createEmbeddingsFaster(df: pd.DataFrame) -> tf.data.Dataset:
    # Load config
    config = loadConfig()

    # Download YAMNET
    yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = hub.load(yamnet_model_handle)

    # Add full path name
    df["full_filename"] = df[["slice_file_name", "fold"]].apply(
        lambda row: formatFilePath(row["fold"], row["slice_file_name"]), axis=1
    )

    results = {"embedding": [], "target": [], "fold": []}
    for _, row in df.iterrows():
        wav, samplingRate = libr.load(
            row["full_filename"], duration=config["DURATION"], sr=YAMNET_SAMPLE_RATE
        )

        scores, embeddings, spectrogram = yamnet_model(wav)
        num_embeddings = tf.shape(embeddings)[0]

        results["embedding"].extend(embeddings.numpy())
        results["target"].extend(np.repeat(row["class"], num_embeddings))
        results["fold"].extend(np.repeat(row["fold"], num_embeddings))

    return pd.DataFrame(results)


def createTransferLearning(
    hiddenLayers: List[int], dropout: float = 0.0, regularization=None, numClasses=10
):
    layers = [
        tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name="input_embedding")
    ]

    for size in hiddenLayers:
        layers.append(
            tf.keras.layers.Dense(
                size, activation="relu", kernel_regularizer=regularization
            )
        )
        layers.append(tf.keras.layers.Dropout(dropout))

    layers.append(tf.keras.layers.Dense(numClasses, activation="softmax"))

    return tf.keras.Sequential(
        layers,
        name="classifier",
    )
