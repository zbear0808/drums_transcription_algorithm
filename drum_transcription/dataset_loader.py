import os
import random
from pathlib import Path
import numpy as np
import os
import logging

log = logging.getLogger(__name__)

class DataSetLoader:
    """
    Class to provide the randomised path names of the x and y samples.

    In addition to this it will provide the path in a way that the dataset
    is balanced being the number of background samples (not containing a instrument
    hit) and foreground samples (containing instrument hit) the same.
    """

    def __init__(self, path_dataset):

        self.path_dataset = self._is_valid_dir(path_dataset)

        self.x_dataset = self._is_valid_dir(os.path.join(path_dataset, "x"))
        self.y_dataset = self._is_valid_dir(os.path.join(path_dataset, "y"))
        self.file_names = self._is_valid_file_names(self.x_dataset, self.y_dataset)

        _temp_x = np.load(self.x_dataset / self.file_names[0])
        _temp_y = np.load(self.y_dataset/ self.file_names[0])
        self.dim_x = _temp_x.shape
        self.dim_y = _temp_y.shape

    def _is_valid_file_names(self, x_data_set, y_dataset):
        """
        Returns the valid names of the files
        :param x_data_set:
        :param y_dataset:
        :return:
        """

        x_names = os.listdir(x_data_set)
        names = []
        for name in x_names:
            if os.path.isfile(y_dataset/name):
                names.append(name)

        random.shuffle(names)
        return names

    def _is_valid_dir(self, path):
        """
        Checks the validity of the directory
        :param path:
        :return:
        """

        if os.path.isdir(path):
            return Path(path)
        else:
            raise ValueError(f"The following path "+ path + "does not exists!")

    def get_training_validation_file_names(self, training_percent: float):
        """
        Returns two list of training samples and validation samples names
        :param training_percent: valure from 0 to 1
        :return:
        """
        log.info("Obtaining samples path. This may take a few minutes...")

        training_number = int(len(self.file_names) * training_percent)

        training_samples_name = self.file_names[:training_number]
        validations_samples_name = self.file_names[training_number:]
        return training_samples_name, validations_samples_name




