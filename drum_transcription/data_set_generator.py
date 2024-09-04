import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    """Generates Data for Keras"""

    def __init__(self, path_x, path_y, files_name, batch_size, dim_x, dim_y,  num_channels=1, shuffle = True):
        """
        Constructor for the class

        :param path_x: path to find the xs
        :param path_y: path to find the ys
        :param files_name: name_files
        :param batch_size:
        :param dim_x:
        :param dim_y:
        :param num_channels:
        :param shuffle:
        """
        self.path_x = path_x
        self.path_y = path_y
        self.files_name = files_name
        self.list_index = np.arange(len(files_name))
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_channels = num_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""

        return int(len(self.files_name) // self.batch_size)

    def on_epoch_end(self):
        """
        Updates index after each epoch

        It is triggered on the very beginning and at the end of each epoch.
        """
        if self.shuffle == True:
            np.random.shuffle(self.list_index)

    def __data_generation(self, temp_file_names):
        """
        Generates the data containing batch_size_samples

        :return: (X, Y) for a batch to be fed to Keras
        """
        X = np.empty((self.batch_size, *self.dim_x, self.num_channels))
        Y = np.empty((self.batch_size, *self.dim_y))

        for i, file_name in enumerate(temp_file_names):
            x = np.load(self.path_x / file_name)
            x = x[..., np.newaxis]
            X[i,:] = x
            Y[i,] = np.load(self.path_y / file_name)

        return X, Y

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.list_index[index * self.batch_size: (index + 1) * self.batch_size]

        list_ID_temp = [self.files_name[k] for k in indexes]

        X, Y = self.__data_generation(list_ID_temp)
        return X, Y