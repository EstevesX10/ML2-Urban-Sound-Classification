import numpy as np
import pandas as pd
import ast
from keras.src.utils import (to_categorical)
from sklearn.model_selection import (train_test_split)
from sklearn.preprocessing import (LabelEncoder)
from sklearn.discriminant_analysis import (StandardScaler)
from sklearn.model_selection import (PredefinedSplit, cross_validate)
from sklearn.pipeline import (Pipeline)
from scikeras.wrappers import (KerasClassifier)

class AudioDataset():
    def __init__(self, datasetFilePath) -> None:
        self.df = pd.read_csv(datasetFilePath)
        self.prepareData()

    def prepareData(self):
        # [TODO] Alter to the proper features
        # Fetch the Features and properly convert them
        self.x = np.array(self.df['MFCC'].apply(lambda x: ast.literal_eval(x)).tolist())

        # Encode target labels
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.df['target'])
        self.y = to_categorical(self.y, num_classes=10)

        # Define a vector for the dataset indices
        indices = np.arange(len(self.x))

        # Train-validation split
        self.train_indices, self.test_indices = train_test_split(indices, stratify=self.y, random_state=0)
        self.trainX, self.trainY = self.x[self.train_indices], self.y[self.train_indices]
        self.testX, self.testY = self.x[self.test_indices], self.y[self.test_indices]


class NetworkPipeline():
    def __init__(self, model=None, epochs=10, batch_size=12, callbacks=[], verbose=0) -> None:
        
        # Check if a model was given
        if model is None:
            raise ValueError("Missing a Model!")
        
        # Create a Keras Classifier
        self.classifier = KerasClassifier(
            build_fn=lambda:model,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Create a Pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),   # Feature scaling
            ('model', self.classifier)      # TensorFlow model
        ])

    def performCV(self, dataset:AudioDataset):
        # Define a Split
        split = PredefinedSplit(dataset.df.iloc[dataset.train_indices]["fold"])

        # Perform Cross Validation
        cv_results = cross_validate(self.pipeline, dataset.trainX, dataset.trainY, cv=split)
        # history = pipeline.fit(trainX, trainY)
        history = cv_results.best_estimator_.named_steps["model"].history_

        return history