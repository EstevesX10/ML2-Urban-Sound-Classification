from typing import Tuple
import numpy as np
import pandas as pd
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, StandardScaler, MinMaxScaler

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


class UrbanSound8kManager:
    def __init__(
        self, dataDimensionality: str = "1D", pathsConfig: dict = None
    ) -> None:
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

        self.dataDimensionality = dataDimensionality

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
        if self.dataDimensionality == "1D":
            fileType = "1D-Processed-MFCCs"

        elif self.dataDimensionality == "2D":
            fileType = "2D-Raw-MFCCs"

        elif self.dataDimensionality == "transfer":
            fileType = "yamnet"

        else:
            # Invalid Data Dimensionality
            raise ValueError(
                "Invalid Data Dimensionality Selected! (Please choose from ['1D', '2D'])"
            )

        # Create a dataframe with all the collected data across all folds
        df = None

        # Iterate through the datasets' folds
        if self.dataDimensionality == "transfer":
            df = pd.read_pickle(self.pathsConfig["Datasets"]["transfer"])
        else:
            for fold in range(1, 11):
                # Load the current fold dataframe
                fold_df = pd.read_pickle(
                    self.pathsConfig["Datasets"][f"Fold-{fold}"][fileType]
                )

                # If the DataFrame has yet to be created, then we initialize it
                if df is None:
                    df = fold_df
                else:
                    # Concatenate the current fold's DataFrame
                    df = pd.concat([df, fold_df], axis=0, ignore_index=True)

        return df

    def getTrainTestSplitFold(
        self, testFold: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        # Calculate the amount of unique target labels
        numClasses = np.unique(df["target"]).size

        # Separate the data into train and test
        train_df = df.loc[df["fold"] != testFold]
        test_df = df.loc[df["fold"] == testFold]

        # Reset indexes
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Binarize target column on the train set and transform the one on the test set
        labelBinarizer = LabelBinarizer()
        trainBinarizedTarget = labelBinarizer.fit_transform(train_df["target"])
        testBinarizedTarget = labelBinarizer.transform(test_df["target"])

        # Update train and test DataFrames with the Binarized Target
        train_df = pd.concat(
            [
                train_df.drop(columns=["target"]),
                pd.DataFrame(trainBinarizedTarget, columns=labelBinarizer.classes_),
            ],
            axis=1,
        )
        test_df = pd.concat(
            [
                test_df.drop(columns=["target"]),
                pd.DataFrame(testBinarizedTarget, columns=labelBinarizer.classes_),
            ],
            axis=1,
        )

        # Evaluate the kind of data dimensionality provided and adapt the method to it
        if self.dataDimensionality == "1D":
            # Define the columns of the features and the target
            featuresCols = train_df.columns[2 : len(train_df.columns) - numClasses]
            targetCols = train_df.columns[-numClasses:]

            # Normalize the data
            standardScaler = StandardScaler()
            # standardScaler = MinMaxScaler(feature_range=(0, 1))

            # Fit the scaler and transform the training data
            train_df[featuresCols] = standardScaler.fit_transform(
                train_df[featuresCols]
            )

            # Transform the test set according to the trained scaler
            test_df[featuresCols] = standardScaler.transform(test_df[featuresCols])

            # Split the data into X and y for both train and test sets
            X_train = train_df[featuresCols].to_numpy()
            y_train = train_df[targetCols].to_numpy()

            X_test = test_df[featuresCols].to_numpy()
            y_test = test_df[targetCols].to_numpy()

        elif self.dataDimensionality == "2D":
            # Define the columns of the features and the target
            featuresCols = "MFCC"
            targetCols = train_df.columns[-numClasses:]

            # Split the data into X and y for both train and test sets
            X_train_ = train_df[featuresCols]
            y_train = train_df[targetCols].to_numpy()

            X_test_ = test_df[featuresCols]
            y_test = test_df[targetCols].to_numpy()

            X_train = np.stack(X_train_.values)
            X_test = np.stack(X_test_.values)

            # Approach 1
            # mean_time_step = X_train.mean(axis=1, keepdims=True)
            # std_time_step = X_train.std(axis=1, keepdims=True)
            # X_train = (X_train - mean_time_step) / std_time_step

            # mean_time_step = X_test.mean(axis=1, keepdims=True)
            # std_time_step = X_test.std(axis=1, keepdims=True)
            # X_test = (X_test - mean_time_step) / std_time_step

            # Approach 2
            # scaler = MinMaxScaler()
            # # Flatten array for scaling, then reshape back
            # X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            # X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        elif self.dataDimensionality == "transfer":
            # Define the columns of the features and the target
            featuresCols = "embedding"
            targetCols = train_df.columns[-numClasses:]

            # Split the data into X and y for both train and test sets
            X_train = train_df[featuresCols]
            y_train = train_df[targetCols].to_numpy()

            X_test = test_df[featuresCols]
            y_test = test_df[targetCols].to_numpy()

            X_train = np.stack(X_train)
            X_test = np.stack(X_test)

        else:
            raise ValueError(
                "[SOMETHING WENT WRONG] Invalid Data Dimensionality Selected!"
            )

        # Return the sets computed
        return X_train, y_train, X_test, y_test

    def cross_validate(self, compiled_model, epochs: int = 100, callbacks: list = None):
        histories = []

        for testFold in range(1, 3):
            X_train, y_train, X_val, y_val = self.getTrainTestSplitFold(
                testFold=testFold
            )

            history = compiled_model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
            )

            histories.append(history)

        return histories
