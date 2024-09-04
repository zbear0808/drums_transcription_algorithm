from gc import callbacks
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GRU, BatchNormalization, Dropout
from tensorflow.keras import Model
from drum_transcription.dataset_loader import DataSetLoader
from drum_transcription.data_set_generator import DataGenerator
from drum_transcription.settings.settings import Settings
from datetime import datetime
import numpy as np
import os
from pathlib import Path

settings = Settings()
path_repo = Path(__file__).parent.parent

class MLModel(Model):
    """ML Model for the algorithm"""

    def __init__(self, x_dim, y_dim):
        """Model constructor"""
        super(MLModel, self).__init__()

        self.conv1 = Conv2D(
            filters=32,
            kernel_size=(3,3),
            padding = 'same',
            activation= 'relu',
        )
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding = 'same', 
            activation= 'relu'
        )
        self.batch_norm2 = BatchNormalization()
        self.max_pool_layer1 = MaxPool2D(
            pool_size=(2, 1)
        )

        self.conv3 = Conv2D(
            filters =64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )
        self.batch_norm3 = BatchNormalization()
        self.conv4 = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation= 'relu'
        )

        self.batch_norm4 = BatchNormalization()
        self.max_pool_layer2 = MaxPool2D(
            pool_size=(2, 1)
        )

        self.flatten_1 = Flatten()
        self.dense1 = Dense(256, activation='relu')
        self.dropout1 = Dropout(settings.dropout)
        self.dense2 = Dense(y_dim[0], activation='sigmoid')

    def call(self, x, training=False):
        """Forward the model"""

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.max_pool_layer1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch_norm2(x)
        x = self.max_pool_layer2(x)
        x = self.flatten_1(x)
        x = self.dense1(x)
        if training:
          x = self.dropout1(x, training=training)
        x = self.dense2(x)
        return x


def train_model(path_dataset):

    batch_size = settings.batch_size
    data_set_loader = DataSetLoader(path_dataset)
    training_samples_name, validations_samples_name = data_set_loader.get_training_validation_file_names(training_percent=0.7)

    training_generator = DataGenerator(
        path_x=data_set_loader.x_dataset,
        path_y= data_set_loader.y_dataset,
        files_name = training_samples_name,
        batch_size = batch_size,
        dim_x = data_set_loader.dim_x,
        dim_y = data_set_loader.dim_y
    )

    validation_generator = DataGenerator(
        path_x=data_set_loader.x_dataset,
        path_y= data_set_loader.y_dataset,
        files_name = validations_samples_name,
        batch_size = batch_size,
        dim_x = data_set_loader.dim_x,
        dim_y = data_set_loader.dim_y
    )
    
    model = MLModel(x_dim=data_set_loader.dim_x, y_dim=data_set_loader.dim_y)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)]
    )

    datetime_res=str(datetime.now())
    datetime_res = datetime_res.replace(':', '.')
    check_point_folder = str(path_repo / "model_checkpoints" / ("checkpoint_" + datetime_res))
    os.makedirs(check_point_folder)
    filepath_cb = check_point_folder + "/chckpnt-{epoch:02d}-{loss:.2f}.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath_cb, monitor='loss', 
                                                   verbose=0, 
                                                   save_best_only=False,
                                                   save_weights_only=False, 
                                                   mode='auto', 
                                                   save_freq='epoch')

    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        steps_per_epoch=len(training_samples_name) // settings.batch_size,
        epochs=settings.n_epoch,
        callbacks=[checkpoint_cb]
    )
    path_model=str(path_repo / "model_full_export")
    model.save(path_model, save_format=tf, overwrite=True)
