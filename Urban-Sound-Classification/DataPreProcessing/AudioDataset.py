import numpy as np
import pandas as pd
import os
import ast
from keras.src.utils import (to_categorical)
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import (LabelEncoder)

class AudioDataset():
    def __init__(self, fold:int=None, intervalStep:int=None, pathsConfig:dict=None) -> None:
        
        # Check if the fold was passed on
        if fold is None:
            raise ValueError("Missing the Dataset's fold!")
        
        # Check if the fold is valid
        if (fold < 0 or fold > 10):
            raise ValueError("Invalid Fold!")

        # Verify if the paths configuration was given
        if pathsConfig is None:
            raise ValueError("Missing a Configuration Dictionary with the paths used in the Project!")

        # Set a Default Value to the intervalStep
        intervalStep = 5 if intervalStep is None else intervalStep

        # Save the dataset fold
        self.fold = fold

        # Step in which we are going to consider the segments of the data
        self.step =  intervalStep 

        # Save the config with important paths
        self.pathsConfig = pathsConfig

        # Load the dataset with the raw features and select the important columnhs
        self.df = pd.read_csv(pathsConfig['Datasets'][f'Fold-{fold}']['Total-Features'])
        self.cols = self.df.columns[2:len(self.df.columns) - 2]
        
        # Create a attribute for the processed dataframe
        self.processed_df = None
        
        # Process the Raw Data
        self.processData()

        # Create a Partition into X and y
        self.createPartition()

    def getFeaturesDetails(self) -> list[dict]:
        """
        # Description
            -> Based on the extracted features from the first audio sample, it computes the amount of
            components we need to use to partition each feature while considering the possible residue.
        -----------------------------------------------------------------------------------------------
        := return: A list with all the data regarding the data formating on the next step.
        """

        # List to store the details of the columns to process
        columnsDetails = []

        # Iterate over the features of the DataFrame
        for feature in self.cols:
            # Analyse the shape of the array of the current feature
            length = len(ast.literal_eval(self.df.iloc[0][feature]))
            # print(length)

            # Calculate the number of components for the current features
            numComponents = length // self.step
            residueSize = length / self.step

            # Update the initial list with the computed data
            columnsDetails.append({
                'feature':feature.replace('-', '_').replace(' ', '_'),
                'totalComponents':numComponents,
                'residueSize':residueSize
            })
        
        return columnsDetails

    def processData(self) -> None:
        """
        # Description
            -> This Method allows to process the previously extracted data into
            multiple components based on a small partitions of the data and a couple metrics.
        -------------------------------------------------------------------------------------
        := return: None, since we are only updating one attribute of the class.
        """

        # If the DataFrame with the processed data has not been computed
        if not os.path.exists(self.pathsConfig['Datasets'][f'Fold-{self.fold}']['Processed-Features']):
            # Fetch the Column's Details
            columnDetails = self.getFeaturesDetails()

            # Iterate row by row and process each extracted vector to get mean, std, ... to obtain multiple columns [MEAN_F1, STD_F1, MEAN_F2, STD_F2, ...]
            for index, row in self.df.iterrows():
                
                # Create a new dictionary for a new line in the Dataframe
                audioSampleData = {}

                # Iterate through the features
                for featureIdx, feature in enumerate(self.cols):

                    # Fetch and Convert the array in the current cell
                    featureArray = np.array(ast.literal_eval(row[feature]))

                    # Create the components
                    for currentComponent in range(1, columnDetails[featureIdx]['totalComponents'] + 1):
                        if currentComponent == columnDetails[featureIdx]['totalComponents'] - 1: 
                            audioSampleData.update({
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Mean":np.mean(featureArray[currentComponent*self.step:]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Median":np.median(featureArray[currentComponent*self.step:]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Std":np.std(featureArray[currentComponent*self.step:])
                            })
                        else:
                            audioSampleData.update({
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Mean":np.mean(featureArray[(currentComponent - 1)*self.step:currentComponent*self.step]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Median":np.median(featureArray[(currentComponent - 1)*self.step:currentComponent*self.step]),
                                f"{columnDetails[featureIdx]['feature']}_{currentComponent}_Std":np.std(featureArray[(currentComponent - 1)*self.step:currentComponent*self.step])
                            })

                # Add the target Label
                audioSampleData.update({
                    'target':row['target']
                })

                # Check if we already have a DataFrame
                if self.processed_df is None:
                    # Create a new one from zero
                    self.processed_df = pd.DataFrame([audioSampleData])
                else:
                    # Create a new DataFram with the new processed audio entry
                    newLine = pd.DataFrame([audioSampleData])

                    # Concatenate the new DataFrame with the previous one
                    self.processed_df = pd.concat([self.processed_df, newLine], ignore_index=True)
            
            # Save the Processed data
            self.processed_df.to_csv(self.pathsConfig['Datasets'][f'Fold-{self.fold}']['Processed-Features'], sep=',', index=False)
        
        # The Data has already been computed
        else:
            # Load the DataFrame
            self.processed_df = pd.read_csv(self.pathsConfig['Datasets'][f'Fold-{self.fold}']['Processed-Features'])

    def createPartition(self) -> None:
        """
        # Description
            -> This method ensures a separation between the dataset features and 
            target labels alongside a proper label encoding and data partitioning.
        --------------------------------------------------------------------------
        := return: None, since we are only updating one attribute of the class.
        """
        
        # Get features
        self.x = self.processed_df[self.processed_df.columns[:-1]].to_numpy()

        # Encode target labels
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.processed_df['target'])
        self.y = to_categorical(self.y, num_classes=10)

        # print(self.y.shape)

        # Define a vector for the dataset indices
        indices = np.arange(len(self.x))

        # Train-validation split
        self.train_indices, self.test_indices = train_test_split(indices, stratify=self.y, random_state=0)
        self.trainX, self.trainY = self.x[self.train_indices], self.y[self.train_indices]
        self.testX, self.testY = self.x[self.test_indices], self.y[self.test_indices]
