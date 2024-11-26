from typing import Tuple
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Input, BatchNormalization, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, LeakyReLU, SpatialDropout2D, GlobalAveragePooling2D  # type: ignore


class CNN(keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # Architecture used in the "An Analysis of Audio Classification Techniques using Deep Learning Architectures" Paper
    def create2DCNN(
        self, input_shape: Tuple, testNumber: int, numClasses: int = 10
    ) -> keras.Sequential:
        if testNumber == 1:
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Conv2D(64, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Conv2D(128, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )
        elif testNumber == 2:
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )
        elif testNumber == 3:
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )
        elif testNumber == 4:
            return keras.Sequential(
                [
                    Input(shape=input_shape),

                    Conv2D(32, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Conv2D(64, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Conv2D(128, (3, 3), activation="relu", padding="same"),
                    BatchNormalization(),
                    MaxPooling2D((2, 2)),

                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.6),
                    Dense(units=numClasses, activation="softmax"),
                ],
                name=f"2D-CNN-V{testNumber}",
            )

        # return keras.Sequential([
        #     Input(shape=input_shape),
        #     Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Dropout(rate=0.2),
        #     Flatten(),
        #     Dense(units=512, activation='relu'),
        #     Dropout(rate=0.2),
        #     Dense(units=1024, activation='relu'),
        #     Dropout(rate=0.2),
        #     Dense(units=2048, activation='relu'),
        #     Dense(units=numClasses, activation='softmax')
        # ])

        # return keras.Sequential([
        #     Input(shape=input_shape),
        #     Conv2D(filters=32, kernel_size=(2, 5), activation=LeakyReLU(alpha=0.1)),
        #     BatchNormalization(),
        #     SpatialDropout2D(rate=0.07),

        #     Conv2D(filters=32, kernel_size=(2, 5), activation=LeakyReLU(alpha=0.1)),
        #     BatchNormalization(),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     SpatialDropout2D(rate=0.07),

        #     Conv2D(filters=64, kernel_size=(2, 5), activation=LeakyReLU(alpha=0.1)),
        #     BatchNormalization(),
        #     SpatialDropout2D(rate=0.14),

        #     Conv2D(filters=64, kernel_size=(2, 5), activation=LeakyReLU(alpha=0.1)),
        #     BatchNormalization(),
        #     GlobalAveragePooling2D(),

        #     Dense(units=numClasses, activation='softmax')
        # ])
