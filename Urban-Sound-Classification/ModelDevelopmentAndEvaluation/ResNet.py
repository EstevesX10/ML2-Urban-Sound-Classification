import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Layer # type: ignore
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization # type: ignore
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.layers import Add, ReLU, Dense # type: ignore

class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, stride=1):
        super().__init__()
        self.stride = stride
        self.filters = filters

        # First Convolutional Layer with Batch Normalization and Relu
        self.conv1 = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
        )
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        self.conv2 = Conv2D(
            filters=filters, kernel_size=3, padding="same", strides=stride
        )
        self.bn2 = BatchNormalization()

        # Define the skip connection
        self.skip_connection = None

        self.add = Add()
        self.relu2 = ReLU()

    def build(self, input_shape):
        # Define the skip connection
        if self.stride != 1 or input_shape[-1] != self.filters:
            self.skip_connection = keras.Sequential(
                [
                    Conv2D(
                        filters=self.filters,
                        kernel_size=1,
                        strides=self.stride**2,
                        padding="same",
                        use_bias=False,
                    ),
                    BatchNormalization(),
                ]
            )
        else:
            # Identity function
            self.skip_connection = lambda x, training: x

    def call(self, x: tf.Tensor, training=False):
        # Save x for the skip connection
        residue = self.skip_connection(x, training=training)

        # Pass the x vector through the previously defined layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Converge the result from the previous layers to the initial residue
        x = self.add([x, residue])
        x = self.relu2(x)

        return x


class ResNet(tf.keras.Model):
    def __init__(self, input_shape, num_classes=10):
        super().__init__()

        self.inputLayer = Input(shape=input_shape)

        self.conv = Conv2D(
            filters=64, kernel_size=7, strides=2, padding="same", use_bias=False
        )
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.pool = MaxPooling2D(pool_size=3, strides=2, padding="same")

        # Define multiple blocks
        self.layer1 = self._buildLayer(filters=64, blocks=2, stride=1)
        self.layer2 = self._buildLayer(filters=128, blocks=2, stride=2)
        self.layer3 = self._buildLayer(filters=256, blocks=2, stride=2)
        self.layer4 = self._buildLayer(filters=512, blocks=2, stride=2)

        self.globalAvgPool = GlobalAveragePooling2D()
        self.fullyConnectedLayer = Dense(num_classes, activation="softmax")

    def _buildLayer(self, filters, blocks, stride):
        # Create a list for the residual layers
        residualLayers = []

        # Create and append the initial residual block
        residualLayers.append(ResidualBlock(filters=filters, stride=stride))

        # Get all the remaining residual blocks
        for _ in range(1, blocks):
            residualLayers.append(ResidualBlock(filters=filters, stride=1))

        # Grab all the residual layers and convert them into a Sequencial Model
        return keras.Sequential(residualLayers)

    def createResNet(self):
        return keras.Sequential([
            # Initial layers
            self.inputLayer,
            self.conv,
            self.bn,
            self.relu,
            self.pool,

            # Residual blocks
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,

            # Apply Max Pool and apply a fully connected layer
            self.globalAvgPool,
            self.fullyConnectedLayer
        ])
