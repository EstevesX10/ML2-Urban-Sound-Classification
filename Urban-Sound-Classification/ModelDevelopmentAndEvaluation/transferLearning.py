import numpy as np
import pandas as pd
import librosa as libr
from Utils.Configuration import loadConfig, loadPathsConfig
from DataPreProcessing.AudioManagement import formatFilePath
from typing import Tuple

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio


class Foo:
    def __init__(self):
        yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
        yamnet_model = hub.load(yamnet_model_handle)

        self.yamnet_model = yamnet_model

        self.classifier = tf.keras.Sequential(
            [
                tf.keras.layers.Input(
                    shape=(1024,), dtype=tf.float32, name="input_embedding"
                ),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(10),
            ],
            name="classifier",
        )

        self.classifier.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"],
        )

    def createDataset(
        self, df: pd.DataFrame
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        config = loadConfig()

        # Add full path name
        df["full_filename"] = df[["slice_file_name", "fold"]].apply(
            lambda row: formatFilePath(row["fold"], row["slice_file_name"]), axis=1
        )

        filenames = df["full_filename"]
        targets = df["class"]
        folds = df["fold"]

        main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

        # Read audio from file
        def load_wav(filename, label, fold):
            file_contents = tf.io.read_file(filename)
            wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
            wav = tf.squeeze(wav, axis=-1)
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            wav = tfio.audio.resample(
                wav, rate_in=sample_rate, rate_out=config["SAMPLE_RATE"]
            )
            return wav, label, fold

        main_ds = main_ds.map(
            load_wav
        )  # , num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        # Extract embeddings from yamnet
        def extract_embedding(wav_data, label, fold):
            scores, embeddings, spectrogram = self.yamnet_model(wav_data)
            num_embeddings = tf.shape(embeddings)[0]
            return (
                embeddings,
                tf.repeat(label, num_embeddings),
                tf.repeat(fold, num_embeddings),
            )

        main_ds = main_ds.map(
            extract_embedding
        ).unbatch()  # , num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).unbatch()

        cached_ds = main_ds.cache()
        train_ds = cached_ds.filter(lambda embedding, label, fold: fold != 10)
        test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 10)

        # remove the folds column now that it's not needed anymore
        remove_fold_column = lambda embedding, label, fold: (embedding, label)

        train_ds = train_ds.map(
            remove_fold_column
        )  # , num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
        test_ds = test_ds.map(
            remove_fold_column
        )  # , num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

        train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

        return train_ds, test_ds
