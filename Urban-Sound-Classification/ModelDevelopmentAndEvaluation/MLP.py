import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import Sequential
from keras.src.layers import (Input, Flatten, Conv1D, Conv2D, MaxPooling1D, BatchNormalization, Dense, Dropout)

class MLP(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def createMLP(self, input_shape, numClasses=10):
        return Sequential([
            Input(shape=input_shape),

            Dense(2048, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(numClasses, activation='softmax')
        ])