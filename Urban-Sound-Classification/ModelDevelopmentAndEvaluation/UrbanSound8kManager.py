from typing import Tuple
import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import confusion_matrix

import keras
from keras.src.callbacks.history import History

from .DataVisualization import plotNetworkTrainingPerformance, plotConfusionMatrix
from .pickleFileManagement import saveObject, loadObject

class UrbanSound8kManager:
    def __init__(
        self, dataDimensionality: str = None, modelType: str = None, pathsConfig: dict = None
    ) -> None:
        """
        # Description
            -> Constructor that helps define new instances of the Class UrbanSound8kManager.
        ------------------------------------------------------------------------------------
        := param: dataDimensionality - Interval considered to segment and process the audio's raw features (previously extracted).
        := param: pathsConfig - Dictionary used to store the paths to important files used throughout the project.
        := return: None, since we are only instanciating a class.
        """

        # Check if a DataDimensionality was given
        if dataDimensionality is None:
            raise ValueError("Missing the Value for Data Dimensionality!")

        # Check if the modelType was passed on
        if modelType is None:
            raise ValueError("Missing the Model Type to be later used for Trainning! [Use \"CNN\", \"MLP\" or \"YAMNET\" - depending on what model you plan to train on the selected data!]")

        # Check if a paths configuration was given
        if pathsConfig is None:
            raise ValueError("Missing a Dictionary with the Paths Configuration!")

        # Save the data dimensionality
        self.dataDimensionality = dataDimensionality

        # Save the type of model we are working with
        self.modelType = modelType

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
            # fileType = "1D-Processed-MFCCs"
            fileType = "1D-Processed-Features"

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
        train_df = df[(df["fold"] != testFold) & (df["fold"] != (testFold % 10 + 1))]
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
                pd.DataFrame(
                    validationBinarizedTarget, columns=labelBinarizer.classes_
                ),
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

            # Split the data into X and y for train, validation and test sets
            X_train = train_df[featuresCols].to_numpy()
            y_train = train_df[targetCols].to_numpy()

            X_val = validation_df[featuresCols].to_numpy()
            y_val = validation_df[targetCols].to_numpy()

            X_test = test_df[featuresCols].to_numpy()
            y_test = test_df[targetCols].to_numpy()

            # Normalize the data
            mean = X_train.mean()
            std = X_train.std()

            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

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
            X_val = (X_val - mean) / std
            X_test = (X_test - mean) / std

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

    def cross_validate(
        self,
        compiledModel: keras.models.Sequential,
        epochs: int = 100,
        callbacks=lambda: [],
    ) -> Tuple[list[History], list[np.ndarray]]:
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
        for testFold in range(1, 11):
            # Partition the data into train and validation
            X_train, y_train, X_val, y_val, X_test, y_test = self.getTrainTestSplitFold(
                testFold=testFold
            )

            # Get the current fold model's file path and history path
            modelFilePath = self.pathsConfig['ModelDevelopmentAndEvaluation'][self.modelType][f"Fold-{testFold}"]["Model"]
            historyFilePath = self.pathsConfig['ModelDevelopmentAndEvaluation'][self.modelType][f"Fold-{testFold}"]["History"]

            # Check if the fold has already been computed
            foldAlreadyComputed = os.path.exists(modelFilePath)

            # Getting the model's current fold path and making sure it exists
            modelFoldPath = Path("/".join(modelFilePath.split('/')[:-1]))
            modelFoldPath.mkdir(parents=True, exist_ok=True)

            # If we have not trained the model, then we need to
            if not foldAlreadyComputed:
                # Train the model
                history = compiledModel.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    callbacks=callbacks(),
                )

                # Save the history
                saveObject(history, filePath=historyFilePath)

                # Save the Model
                saveObject(compiledModel, filePath=modelFilePath)

            else:
                # Load the previously computed fold history and trained model
                history = loadObject(filePath=historyFilePath)
                compiledModel = loadObject(filePath=modelFilePath)

            # Get predictions
            y_pred = np.argmax(compiledModel.predict(X_test), axis=1)
            y_true = np.argmax(y_test, axis=1)

            # Compute confusion matrix
            confusionMatrix = confusion_matrix(y_true, y_pred)

            # Plotting model training performance
            plotNetworkTrainingPerformance(
                confusionMatrix=confusionMatrix,
                trainHistory=history.history,
                targetLabels=self.classes_,
            )

            # If we are training, then we need to set back the initial weights to the network
            if not foldAlreadyComputed:
                # Set back the initial weights
                compiledModel.set_weights(initial_weights)

            # Append results
            histories.append(history)
            confusionMatrices.append(confusionMatrix)

            break
        
        # Compute the global confusion Matrix
        globalConfusionMatrix = confusionMatrices[0]
        for m in confusionMatrices[1:]:
            globalConfusionMatrix += m

        # Plot the Global Confusion Matrix
        plotConfusionMatrix(
            globalConfusionMatrix,
            title="Global Confusion Matrix",
            targetLabels=self.classes_,
        )

        # Return the histories and the confusion matrices
        return histories, confusionMatrices
