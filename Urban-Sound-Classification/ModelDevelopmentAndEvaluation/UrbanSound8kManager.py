from typing import Tuple, Callable
import numpy as np
import pandas as pd
import os
from pathlib import Path

from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from tensorflow import keras
from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import get_custom_objects  # type: ignore

from .DataVisualization import plotNetworkTrainingPerformance, plotConfusionMatrix
from .pickleFileManagement import saveObject, loadObject


class UrbanSound8kManager:
    def __init__(
        self,
        featuresToUse: str = None,
        modelType: str = None,
        testNumber: int = None,
        pathsConfig: dict = None,
    ) -> None:
        """
        # Description
            -> Constructor that helps define new instances of the Class UrbanSound8kManager.
        ------------------------------------------------------------------------------------
        := param: featuresToUse - Features to consider use (from the ones previously extracted).
        := param: modelType - Name of the Model which is going to be used for training.
        := param: testNumber - Number of the test the current model is going to perform.
        := param: pathsConfig - Dictionary used to store the paths to important files used throughout the project.
        := return: None, since we are only instanciating a class.
        """

        # Check if a DataDimensionality was given
        if featuresToUse is None:
            raise ValueError("Missing the Value for Data Dimensionality!")

        # Check if the modelType was passed on
        if modelType is None:
            raise ValueError(
                'Missing the Model Type to be later used for Trainning! [Use "CNN", "MLP" or "YAMNET" - depending on what model you plan to train on the selected data!]'
            )

        # Verify if the test number was given
        if testNumber is None:
            raise ValueError("Missing the Model's test number!")

        # Check if a paths configuration was given
        if pathsConfig is None:
            raise ValueError("Missing a Dictionary with the Paths Configuration!")

        # Save the data dimensionality
        self.featuresToUse = featuresToUse

        # Save the type of model we are working with
        self.modelType = modelType

        # Save the number of the test
        self.testNumber = testNumber

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

        if (
            self.featuresToUse not in self.pathsConfig["Datasets"]["Fold-1"].keys()
            and self.featuresToUse != "transfer"
        ):
            # Invalid Data Dimensionality
            raise ValueError(
                f'Invalid Features Selected! (Please choose from {self.pathsConfig["Datasets"]["Fold-1"].keys()})'
            )

        # Create a dataframe with all the collected data across all folds
        df = None

        # Iterate through the datasets' folds
        if self.featuresToUse == "transfer":
            df = pd.read_pickle(self.pathsConfig["Datasets"][self.featuresToUse])
        else:
            for fold in range(1, 11):
                # Load the current fold dataframe
                fold_df = pd.read_pickle(
                    self.pathsConfig["Datasets"][f"Fold-{fold}"][self.featuresToUse]
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
        if (
            self.featuresToUse == "1D-Processed-MFCCs"
            or self.featuresToUse == "1D-Processed-Features"
        ):
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

        elif self.featuresToUse == "2D-Raw-MFCCs":
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

        elif self.featuresToUse == "transfer":
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

    def getAllFolds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        # Description
            -> This method helps get all the data regarding all folds
        which is going to be used to create the t-SNE plot.
        -------------------------------------------------------------
        := return: X and y sets.
        """

        # Manage data
        df = self.manageData()

        # Evaluate the kind of data dimensionality provided and adapt the method to it
        if self.featuresToUse == "1D-Processed-MFCCs" or self.featuresToUse == "1D-Processed-Features":
            # Define the columns of the features and the target
            featuresCols = df.columns[2:-1]
            targetCols = df.columns[-1:]

            # Split the data into X and y for train, validation and test sets
            X = df[featuresCols].to_numpy()
            y = df[targetCols].to_numpy()

        elif self.featuresToUse == "2D-Raw-MFCCs":
            # Define the columns of the features and the target
            featuresCols = "MFCC"
            targetCols = df.columns[-1:]

            # Split the data into X and y for train, validation and test sets
            X = df[featuresCols]
            y = df[targetCols].to_numpy()

            # Stack the data
            X = np.stack(X)

        elif self.featuresToUse == "transfer":
            # Define the columns of the features and the target
            featuresCols = "embedding"
            targetCols = "target"

            # Split the data into X and y for train, validation and test sets
            X = df[featuresCols]
            y = df[targetCols].to_numpy()

            # Stack the data
            X = np.stack(X)

        else:
            raise ValueError(
                "[SOMETHING WENT WRONG] Invalid Data Dimensionality Selected!"
            )

        return X, y

    def crossValidate(
        self,
        createModel: Callable[[], keras.models.Sequential],
        numberFolds: int = 10,
        epochs: int = 100,
        batchSize: int = 32,
        callbacks=lambda: [],
    ) -> Tuple[list[float], list[History], list[np.ndarray]]:
        """
        # Description
            -> This method allows to perform cross-validation over the UrbanSound8k dataset
            given a compiled Model.
        ------------------------------------------------------------------------------------
        := param: compiledModel - Keras sequential model previously compiled.
        := param: numberFolds - Number of Folds to perform the CV on.
        := param: epochs - Number of iterations to train the model at each fold.
        := param: callbacks - List of parameters that help monitor and modify the behavior of your model during training, evaluation and inference.
        := return: A list with the performance mestrics (History) of the model at each fold.
        """

        assert 0 < numberFolds <= 10, f"invalid number of iterations: {numberFolds}"

        # Initialize a list to store all the model's history for each fold
        histories = []

        # Initialize a list to store all the model's confusion matrices for each fold
        confusionMatrices = []

        # Create a List to store the Folds Accuracies
        foldsBalancedAccuracy = []

        # Perform Cross-Validation
        for testFold in range(1, numberFolds + 1):
            # Create new instance of the Model
            compiledModel = createModel(testNumber=self.testNumber)

            # Partition the data into train and validation
            X_train, y_train, X_val, y_val, X_test, y_test = self.getTrainTestSplitFold(
                testFold=testFold
            )

            # Get the current fold model's file path and history path
            modelFilePath = self.pathsConfig["ModelDevelopmentAndEvaluation"][
                self.modelType
            ][f"Test-{self.testNumber}"][f"Fold-{testFold}"]["Model"]
            historyFilePath = self.pathsConfig["ModelDevelopmentAndEvaluation"][
                self.modelType
            ][f"Test-{self.testNumber}"][f"Fold-{testFold}"]["History"]

            # Check if the fold has already been computed
            foldAlreadyComputed = os.path.exists(modelFilePath)

            # Getting the model's current fold path and making sure it exists
            modelFoldPath = Path("/".join(modelFilePath.split("/")[:-1]))
            modelFoldPath.mkdir(parents=True, exist_ok=True)

            # If we have not trained the model, then we need to
            if not foldAlreadyComputed:
                # Train the model
                history = compiledModel.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batchSize,
                    epochs=epochs,
                    callbacks=callbacks(),
                )

                # Save the history
                saveObject(history, filePath=historyFilePath)

                # Save the Model
                compiledModel.save(modelFilePath)

                # Clear session
                keras.backend.clear_session()

            else:
                # Load the previously computed fold history
                history = loadObject(filePath=historyFilePath)

                # Load the model
                compiledModel = load_model(modelFilePath)

            # Get predictions
            y_pred = np.argmax(compiledModel.predict(X_test), axis=1)
            y_true = np.argmax(y_test, axis=1)

            # Calculate the current fold Accuracy
            currentFoldAccuracy = balanced_accuracy_score(y_true, y_pred)

            # Append the current fold accuracy to the previous list
            foldsBalancedAccuracy.append(currentFoldAccuracy)

            # Compute confusion matrix
            confusionMatrix = confusion_matrix(y_true, y_pred)

            # Plotting model training performance
            plotNetworkTrainingPerformance(
                confusionMatrix=confusionMatrix,
                title=f"[Test-{self.testNumber}] [{self.modelType}] Fold-{testFold}",
                trainHistory=history.history,
                targetLabels=self.classes_,
            )

            # Append results
            histories.append(history)
            confusionMatrices.append(confusionMatrix)

        # Return the histories and the confusion matrices
        return np.array(foldsBalancedAccuracy), histories, confusionMatrices

    def plotGlobalConfusionMatrix(self, confusionMatrices: list[np.ndarray]) -> None:
        """
        # Description
            -> This method helps to compute and display the global confusion matrix.
        ----------------------------------------------------------------------------
        := param: confusionMatrices - List with all the confusion matrices computed throughout all folds.
        := return: None, since we are only plotting a confusion matrix.
        """

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
