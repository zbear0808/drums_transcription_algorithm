import os
from pathlib import Path
import numpy as np
from tensorflow import keras
from drum_transcription.settings.settings import Settings
from drum_transcription import GROOVE_PITCH_POST_PROCESS
import matplotlib.pyplot as plt
from sklearn import metrics

settings = Settings()

repo_path = Path(__file__).parent.parent

def confussion_matrix(data_preprocess_path):
    """
    Computes the confussion matrix given a data_preprocess_path
    :param data_preprocess_path:
    :return:
    """

    path_x = Path(data_preprocess_path) / 'x'
    path_y = Path(data_preprocess_path) / 'y'

    model_full_export = repo_path / "model_full_export"
    model = keras.models.load_model(str(model_full_export))

    y_exp_list = []
    y_pre_list = []
    for file_name in os.listdir(path_x):

        x = np.load(path_x / file_name)
        y_expected = np.load(path_y / file_name)

        y_exp_list.append(y_expected)
        x = x[np.newaxis, ...]
        x = x[..., np.newaxis]

        y_predicted = model.predict(x)
        y_pre_binary = np.where(y_predicted > settings.threshold_prediction, 1, 0)
        y_pre_binary = y_pre_binary[0, :]

        y_pre_list.append(y_pre_binary)

    y_pre_list = np.array(y_pre_list)
    y_exp_list = np.array(y_exp_list)

    cn_matrix = metrics.multilabel_confusion_matrix(
        y_true=y_exp_list,
        y_pred=y_pre_list
    )

    return cn_matrix