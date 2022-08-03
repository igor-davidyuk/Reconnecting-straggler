# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""You may copy this file as the starting point of your own model."""

from time import sleep
from tqdm import tqdm

import tensorflow.keras as ke
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from openfl.federated import KerasTaskRunner


class KerasCNN(KerasTaskRunner):
    """A basic convolutional neural network model."""

    def __init__(self, **kwargs):
        """
        Initialize.

        Args:
            **kwargs: Additional parameters to pass to the function
        """
        super().__init__(**kwargs)

        self.model = self.build_model(self.feature_shape, self.data_loader.num_classes, **kwargs)

        self.initialize_tensorkeys_for_functions()

        self.model.summary(print_fn=self.logger.info)

        if self.data_loader is not None:
            self.logger.info(f'Train Set Size : {self.get_train_data_size()}')
            self.logger.info(f'Valid Set Size : {self.get_valid_data_size()}')

    def build_model(self,
                    input_shape,
                    num_classes,
                    conv_kernel_size=(4, 4),
                    conv_strides=(2, 2),
                    conv1_channels_out=16,
                    conv2_channels_out=32,
                    final_dense_inputsize=100,
                    **kwargs):
        """
        Define the model architecture.

        Args:
            input_shape (numpy.ndarray): The shape of the data
            num_classes (int): The number of classes of the dataset

        Returns:
            tensorflow.python.keras.engine.sequential.Sequential: The model defined in Keras

        """
        model = Sequential()

        model.add(Conv2D(conv1_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu',
                         input_shape=input_shape))

        model.add(Conv2D(conv2_channels_out,
                         kernel_size=conv_kernel_size,
                         strides=conv_strides,
                         activation='relu'))

        model.add(Flatten())

        model.add(Dense(final_dense_inputsize, activation='relu'))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=ke.losses.categorical_crossentropy,
                      optimizer=ke.optimizers.Adam(),
                      metrics=['accuracy'])

        # initialize the optimizer variables
        opt_vars = model.optimizer.variables()

        for v in opt_vars:
            v.initializer.run(session=self.sess)

        return model

    def train(self, col_name, round_num, *args, **kwargs):
        results = super().train(col_name, round_num, *args, **kwargs)

        col_data_path = str(self.data_loader._data_path)

        sleep_time = 10

        # We slow down the second col for 2nd round
        # But all the other rounds it will be faster than the rest
        if col_data_path=='2':
            if round_num == 2: 
                sleep_time = 70
            else:
                sleep_time = 3

        print(F'\n\nHOLDING COL {col_name} DATA_PATH {col_data_path} '
            f'FOR {sleep_time} SECONDS ON ROUND {round_num}\n\n')
        for i in tqdm([0]*sleep_time, desc=f'Holding col {col_data_path}'):
            sleep(1)
        return results