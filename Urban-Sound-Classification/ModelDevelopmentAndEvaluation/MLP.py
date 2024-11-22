from typing import Tuple
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import (Input, Flatten, Conv1D, Conv2D, MaxPooling1D, BatchNormalization, Dense, Dropout) # type: ignore
from tensorflow.keras.regularizers import L2 # type: ignore

class MLP(keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def createMLP(self, input_shape:Tuple, testNumber:int, numClasses:int=10) -> keras.Sequential:
        if testNumber == 1 or testNumber == 2:
            return keras.Sequential([
                Input(shape=input_shape),

                Dense(80, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(40, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(20, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(numClasses, activation='softmax')
            ], name=f"MLP-V{testNumber}")
        
        elif testNumber == 3:
             return keras.Sequential([
                Input(shape=input_shape),

                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),

                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                
                Dense(numClasses, activation='softmax')
            ], name=f"MLP-V{testNumber}")
        
        else:
            raise ValueError("TO BE IMPLEMENTED")

        # return keras.Sequential([
        #     Input(shape=input_shape),

        #     # Dense(2048, activation='relu'),
        #     # BatchNormalization(),
        #     # Dropout(0.2),
            
        #     # Dense(1024, activation='relu'),
        #     # BatchNormalization(),
        #     # Dropout(0.2),
            
        #     Dense(512, activation='relu',kernel_regularizer=L2(1e-4)),
        #     BatchNormalization(),
        #     Dropout(0.2),
            
        #     # Dense(256, activation='relu'),
        #     # BatchNormalization(),
        #     # Dropout(0.2),
            
        #     Dense(128, activation='relu', kernel_regularizer=L2(1e-4)),
        #     BatchNormalization(),
        #     Dropout(0.2),
            
        #     # Dense(64, activation='relu'),
        #     # BatchNormalization(),
        #     # Dropout(0.2),
            
        #     Dense(numClasses, activation='softmax')
        # ])