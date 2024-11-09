import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import Sequential
from keras.src.layers import (Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout)

class CNN(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Architecture used in the "An Analysis of Audio Classification Techniques using Deep Learning Architectures" Paper
    def create2DCNN(self, numClasses=10):
        return Sequential([
            Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=0.2),
            Dense(units=512, activation='relu'),
            Dropout(rate=0.2),
            Dense(units=1024, activation='relu'),
            Dropout(rate=0.2),
            Dense(units=2048, activation='relu'),
            Dense(units=numClasses, activation='softmax')
        ])
    
    def create1DCNN(self, numClasses=10):
        return Sequential([
            Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(rate=0.2),
            Dense(units=512, activation='relu'),
            Dropout(rate=0.2),
            Dense(units=1024, activation='relu'),
            Dropout(rate=0.2),
            Dense(units=2048, activation='relu'),
            Dense(units=numClasses, activation='softmax')
        ])