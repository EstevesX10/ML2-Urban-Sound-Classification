import pandas as pd
import librosa as libr
from Utils.Configuration import loadConfig
from DataPreProcessing.AudioManagement import formatFilePath
import numpy as np
from typing import List

import tensorflow as tf
import tensorflow_hub as hub

YAMNET_SAMPLE_RATE = 16_000


# TODO: we probably dont need this
def createEmbeddings(df: pd.DataFrame) -> tf.data.Dataset:
    # Load config
    config = loadConfig()

    # Download YAMNET
    yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = hub.load(yamnet_model_handle)

    # Add full path name
    df["full_filename"] = df[["slice_file_name", "fold"]].apply(
        lambda row: formatFilePath(row["fold"], row["slice_file_name"]), axis=1
    )

    filenames = df["full_filename"]
    targets = df["class"]
    folds = df["fold"]

    main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

    # Read audio from file
    @tf.py_function(Tout=(tf.float32, tf.string, tf.int64))
    def load_wav(filename, label, fold):
        wav, samplingRate = libr.load(
            filename.numpy(), duration=config["DURATION"], sr=YAMNET_SAMPLE_RATE
        )

        return wav, label, fold

    main_ds = main_ds.map(
        load_wav, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )

    # Extract embeddings from yamnet
    def extract_embedding(wav_data, label, fold):
        scores, embeddings, spectrogram = yamnet_model(wav_data)
        num_embeddings = tf.shape(embeddings)[0]
        return (
            embeddings,
            tf.repeat(label, num_embeddings),
            tf.repeat(fold, num_embeddings),
        )

    main_ds = main_ds.map(
        extract_embedding, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    ).unbatch()

    cached_ds = main_ds.cache()
    return cached_ds

    # train_ds = cached_ds.filter(lambda embedding, label, fold: fold != 10)
    # test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 10)

    # train_ds = train_ds.cache()
    # test_ds = test_ds.cache()

    # remove the folds column now that it's not needed anymore
    # remove_fold_column = lambda embedding, label, fold: (embedding, label)

    # train_ds = train_ds.map(
    #     remove_fold_column, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    # )
    # test_ds = test_ds.map(
    #     remove_fold_column, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    # )

    # train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    # test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    # return train_ds, test_ds


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
    hiddenLayers: List[int], dropout: float = 0.0, numClasses=10
):
    layers = [
        tf.keras.layers.Input(shape=(1024,), dtype=tf.float32, name="input_embedding")
    ]

    for size in hiddenLayers:
        layers.append(tf.keras.layers.Dense(size, activation="relu"))
        layers.append(tf.keras.layers.Dropout(dropout))

    layers.append(tf.keras.layers.Dense(numClasses, activation="softmax"))

    return tf.keras.Sequential(
        layers,
        name="classifier",
    )
