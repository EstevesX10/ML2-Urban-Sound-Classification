from typing import (Tuple)
import numpy as np
import pandas as pd
import os
import ast
from keras.src.utils import (to_categorical)
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import (LabelEncoder, LabelBinarizer)

"""
        # 1D
        # Process the features and target into Arrays
        self.x = processed1D_df[processed1D_df.columns[:-1]].to_numpy()
        self.y = processed1D_df['target']


        # 2D 
        # Get the Raw MFCCs
        rawMFCCs = pd.read_pickle(self.pathsConfig['Datasets'][f'Fold-{self.fold}']['Raw-MFCCs-Feature'])        

        # Compute features
        aux = rawMFCCs['MFCC'].to_numpy()
        # print(np.array(aux[0]).shape, type(aux[0]))
        
        self.x = np.expand_dims(aux[0], axis=0)
        
        for sample in aux[1:]:
            sample = np.expand_dims(sample, axis=0)
            self.x = np.vstack((self.x, sample))

        # Save the target labels
        self.y = rawMFCCs['target']
        """

class UrbanSound8kManager():
    def __init__(self, dataDimensionality:str=None, pathsConfig:dict=None) -> None:
        """
        # Description 
            -> Constructor that helps define new instances of the Class UrbanSound8kManager.
        ------------------------------------------------------------------------------------
        := param: dataDimensionality - Interval considered to segment and process the audio's raw features (previously extracted).
        := param: pathsConfig - Dictionary used to store the paths to important files used throughout the project.
        := return: None, since we are only instanciating a class.
        """
        
        # Check if a paths configuration was given
        if pathsConfig is None:
            raise ValueError("Missing a Dictionary with the Paths Configuration!")

        # Set a Default values to the dataDimensionality and intervalStep
        self.dataDimensionality = '1D' if dataDimensionality is None else dataDimensionality

        # Save the dictionary with the file paths
        self.pathsConfig = pathsConfig

    def manageData(self) -> pd.DataFrame:
        """
        # Description
            -> This method allows a easy management of the data from all the
            collected DataFrames in order to create a DataFrame with all the information.
        ------------------------------------------------------------------------
        := return: Train and Test Pandas DataFrames.
        """

        # Interpret the files to use depending on the data Dimensionality provided
        if self.dataDimensionality == '1D':
            fileType = 'Processed-1D-Features'

        elif self.dataDimensionality == '2D':
            fileType = 'Raw-MFCCs-Feature'

        else:
            # Invalid Data Dimensionality
            raise ValueError("Invalid Data Dimensionality Selected! (Please choose from ['1D', '2D'])")
        
        # Create a dataframe with all the collected data across all folds
        df = None

        # Iterate through the datasets' folds
        for fold in range(1, 11):
            # Load the current fold dataframe
            fold_df = pd.read_pickle(self.pathsConfig['Datasets'][f'Fold-{fold}'][fileType])

            # If the DataFrame has yet to be created, then we initialize it
            if df is None:
                df = fold_df
            else:
                # Concatenate the current fold's DataFrame
                df = pd.concat([df, fold_df], axis=0, ignore_index=True)

        return df

    def getTrainTestSplitFold(self, testFold:int=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        # Description
            -> This method allows to obtain the UrbanSound8k's overall train and test
            sets across all folds considering the one selected as the test fold.
        ---------------------------------------------------------------------------------------------------------------------
        := param: testFold - Dataset's Fold whose data is to be used for testing. [NOTE] testFold must be in [1, 2, ..., 10].
        := return: The train and test sets to be used to perform one of the 10-Fold Cross Validation
        """

        # Check if the testFold is given
        if testFold is None:
            raise ValueError("Missing the number of the Test Fold!")

        # Verify the integrity of the test fold selected
        if testFold < 1 or testFold > 10:
            raise ValueError("Invalid Test Fold!") 

        # Manage data from all the collected DataFrames
        df = self.manageData()

        # Separate the data into train and test
        train_df = df.loc[df['fold'] != testFold]
        test_df = df.loc[df['fold'] == testFold]

        # Reset indexes
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # return train_df, test_df

        # Binarize target column on the train set and transform the one on the test set
        labelBinarizer = LabelBinarizer()
        trainBinarizedTarget = labelBinarizer.fit_transform(train_df['target'])
        testBinarizedTarget = labelBinarizer.transform(test_df['target'])

        # Update train and test DataFrames with the Binarized Target
        train_df = pd.concat([train_df.drop(columns=['target']), pd.DataFrame(trainBinarizedTarget, columns=labelBinarizer.classes_)], axis=1)
        test_df = pd.concat([test_df.drop(columns=['target']), pd.DataFrame(testBinarizedTarget, columns=labelBinarizer.classes_)], axis=1)
        
        return train_df, test_df

        # Convert the binary target output into a DataFrame
        binary_df = pd.DataFrame(binaryTarget, columns=labelBinarizer.classes_)
        
        df = pd.concat([df.drop(columns='')])

        # Define a set for the visited folds
        # visitedFolds = set()
        
        # # Get the test sets
        # testX = self.folds[testFold].x
        # testY = self.folds[testFold].y

        # # If our test fold is the first, then we consider the 2nd fold's data as the beggining of our train sets
        # if testFold == 1:
        #     trainX = self.folds[2].x
        #     trainY = self.folds[2].y
        #     visitedFolds.add(2)
        # # Otherwise, we can start with the first fold data to begin the train set
        # else:
        #     trainX = self.folds[1].x
        #     trainY = self.folds[1].y
        #     visitedFolds.add(1)

        # # Iterate through eachn fold
        # for currentFold in range(1, 3):
        #     # Ignore if the fold corresponds to test or if it has already been visited
        #     if currentFold == testFold or currentFold in visitedFolds:
        #         continue
        #     # Update the train sets with the current fold's training data
        #     else:
        #         trainX = np.concatenate((trainX, self.folds[currentFold].x), axis=0)
        #         trainY = np.concatenate((trainY, self.folds[currentFold].y), axis=0)
        
        # # Return the final train and test sets
        # return trainX, trainY, testX, testY

