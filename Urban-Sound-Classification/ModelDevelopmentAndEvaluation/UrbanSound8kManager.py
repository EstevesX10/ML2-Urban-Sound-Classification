from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import confusion_matrix

import keras
from keras.src.callbacks.history import History

from .DataVisualization import plotNetworkTrainingPerformance

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        # Separate the data into train, validation and test
        train_df = df[(df["fold"] != testFold) & (df["fold"] != (testFold + 1) % 10)]
        validation_df = df[(df["fold"] == (testFold % 10 + 1))]
        test_df = df[(df["fold"] == testFold)]

        # Reset indexes
        train_df = train_df.reset_index(drop=True)
        validation_df = validation_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Binarize target column on the train set and transform the one on the test set
        labelBinarizer = LabelBinarizer()
        trainBinarizedTarget = labelBinarizer.fit_transform(train_df["target"])
        validationBinarizedTarget = labelBinarizer.transform(validation_df["target"])
        testBinarizedTarget = labelBinarizer.transform(test_df["target"])
        self.classes_ = labelBinarizer.classes_

        # Update train, validation and test DataFrames with the Binarized Target
        train_df = pd.concat(
            [
                train_df.drop(columns=["target"]),
                pd.DataFrame(trainBinarizedTarget, columns=labelBinarizer.classes_),
            ],
            axis=1,
        )
        validation_df = pd.concat(
            [
                validation_df.drop(columns=["target"]),
                pd.DataFrame(validationBinarizedTarget, columns=labelBinarizer.classes_),
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

            # Transform the validation and test sets according to the trained scaler
            validation_df[featuresCols] = standardScaler.transform(validation_df[featuresCols])
            test_df[featuresCols] = standardScaler.transform(test_df[featuresCols])

            # Split the data into X and y for train, validation and test sets
            X_train = train_df[featuresCols].to_numpy()
            y_train = train_df[targetCols].to_numpy()

            X_val = validation_df[featuresCols].to_numpy()
            y_val = validation_df[targetCols].to_numpy()

            X_test = test_df[featuresCols].to_numpy()
            y_test = test_df[targetCols].to_numpy()

        elif self.dataDimensionality == "2D":
            # Define the columns of the features and the target
            featuresCols = "MFCC"
            targetCols = train_df.columns[-numClasses:]

            # Split the data into X and y for train, validation and test sets
            X_train = train_df[featuresCols]
            y_train = train_df[targetCols].to_numpy()

            X_val = validation_df[featuresCols]
            y_val = validation_df[targetCols].to_numpy()

            X_test = test_df[featuresCols]
            y_test = test_df[targetCols].to_numpy()

            # Stack the data
            X_train = np.stack(X_train)
            X_val = np.stack(X_val)
            X_test = np.stack(X_test)

            # Normalize the data
            mean = X_train.mean()
            std = X_train.std()

            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

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

            # Split the data into X and y for train, validation and test sets
            X_train = train_df[featuresCols]
            y_train = train_df[targetCols].to_numpy()

            X_val = validation_df[featuresCols]
            y_val = validation_df[targetCols].to_numpy()

            X_test = test_df[featuresCols]
            y_test = test_df[targetCols].to_numpy()

            X_train = np.stack(X_train)
            X_val = np.stack(X_val)
            X_test = np.stack(X_test)

        else:
            raise ValueError(
                "[SOMETHING WENT WRONG] Invalid Data Dimensionality Selected!"
            )

        # Return the sets computed
        return X_train, y_train, X_val, y_val, X_test, y_test

    def cross_validate(self, compiledModel:keras.models.Sequential, epochs: int = 100, callbacks: list = None) -> Tuple[list[History], list[np.ndarray]]:
        """
        # Description
            -> This method allows to perform cross-validation over the UrbanSound8k dataset
            given a compiled Model.
        ------------------------------------------------------------------------------------
        := param: compiledModel - Keras sequential model previously compiled.
        := param: epochs - Number of iterations to train the model at each fold.
        := param: callbacks - List of parameters that help monitor and modify the behavior of your model during training, evaluation and inference.
        := return: A list with the performance mestrics (History) of the model at each fold.
        """
        
        # Initialize a list to store all the model's history for each fold
        histories = []

        # Initialize a list to store all the model's confusion matrices for each fold
        confusionMatrices = []

        # Geting the model initial weights
        initial_weights = compiledModel.get_weights()

        # Perform Cross-Validation
        for testFold in tqdm(range(1, 11), desc="Cross-validating..."):
            # Partition the data into train and validation
            X_train, y_train, X_val, y_val, X_test, y_test = self.getTrainTestSplitFold(
                testFold=testFold
            )

            # Train the model
            history = compiledModel.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
            )

            # Get predictions
            y_pred = np.argmax(compiledModel.predict(X_test), axis=1)
            y_true = np.argmax(y_test, axis=1)

            # Compute confusion matrix
            confusionMatrix = confusion_matrix(y_true, y_pred)

            # Plotting model training performance
            plotNetworkTrainingPerformance(
                confusionMatrix=confusionMatrix, trainHistory=history.history, targetLabels=self.classes_
            )

            # Set back the initial weights
            compiledModel.set_weights(initial_weights)

            # Append results
            histories.append(history)
            confusionMatrices.append(confusionMatrix)

        return histories, confusionMatrices
