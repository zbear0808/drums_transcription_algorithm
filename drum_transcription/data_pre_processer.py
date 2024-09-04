"""DataPreProcesserclass"""
from drum_transcription.settings.settings import Settings
import librosa.feature as feature
from drum_transcription.pre_processing import MidiMessages
from drum_transcription.pre_processing import LabeleledFeatures
from drum_transcription.pre_processing import Labels
from drum_transcription import GROOVE_PITCH_NAMES
from pathlib import Path
import numpy as np
import os
import logging
import librosa

log = logging.getLogger(__name__)


class DataPreProcesser:
    """This class will take as arguments  file and pre-process the data"""

    settings = Settings()

    def __init__(self, data_set_path: str, output_path: str, recursive: bool = True):
        """Constructor for the DataPreProcesser"""

        self.files_location: List[str] = []
        self.output_path = output_path

        self.y_output = os.path.join(output_path, "y")
        self.x_output = os.path.join(output_path, "x")
        if not os.path.isdir(output_path):
            os.makedirs(self.y_output)
            os.makedirs(self.x_output)
        else:
            logging.warning("the path " + output_path + " is already in use")

        self.discover_dataset_locations(data_set_path, recursive=recursive)
        log.info("Discovered " + str(len(self.files_location)) + " dataset locations!")

        self.settings = Settings()

    def discover_dataset_locations(self, path, recursive: bool = True):
        """
        It will update the test file folders location recursively

        :param content_path:
        :return: if the algorithm does it recursively
        """
        dir_files_name = os.listdir(path)
        for filename in dir_files_name:
            full_path = Path(os.path.join(path, filename))

            if full_path.suffix in [".mid", ".midi", ".wav"]:
                self.files_location.append(full_path.parent)
                break

            elif recursive and os.path.isdir(full_path):
                self.discover_dataset_locations(full_path, recursive)

    def pre_processing(self, wav_file, midi_file):
        """
        It pre-process the data it will return a list with features and labels on onset times

        :param wav_file:
        :param midi_file:
        :return:
        """

        y, sr = librosa.load(wav_file, sr=None)
        y_size = len(y)
        labelled_features = []

        if sr != self.settings.sampling_frequency:
            log.warning(
                "The sampling frequency of the audio " + wav_file + " is not " + str(self.settings.sampling_frequency)
            )

        onset_idx = librosa.onset.onset_detect(
            y,
            sr=self.settings.sampling_frequency,
            hop_length=self.settings.hop_size_onset,
            units="samples"
        )

        samples_window = int(self.settings.analysis_window_s * self.settings.sampling_frequency)
        midi_messages = MidiMessages(midi_file)
        for onset in onset_idx:
            onset_t0 = int(onset - samples_window/4)
            onset_tf = onset_t0 + samples_window
            if onset_tf >= y_size or onset_t0 < 0:
                continue


            window = y[onset_t0: onset_tf]

            mfcc_features = feature.mfcc(
                y=window,
                sr=self.settings.sampling_frequency,
                hop_length=self.settings.FFT_hop_size,
                n_mfcc=self.settings.n_mfcc_features
            )

            # normalize features
            mfcc_features = (mfcc_features - np.min(mfcc_features))/ (np.max(mfcc_features) - np.min(mfcc_features))

            t_0 = onset_t0 / self.settings.sampling_frequency
            t_f = onset_tf / self.settings.sampling_frequency
            notes = list(midi_messages.get_query_midi(t_0, t_f))

            label = Labels()
            [label.element_at(GROOVE_PITCH_NAMES[note][0]) for note in notes]

            if not label.is_empty():
                labelled_features.append(
                    LabeleledFeatures(
                        t_initial = onset / self.settings.sampling_frequency,
                        features = mfcc_features,
                        labels = label
                    )
                )

        return labelled_features


    def export_data(self, labelled_features, track_number):
        """
        We export the data

        :param labelled_features:
        :param track_number:
        :return:
        """

        x_folder = Path(self.x_output)
        y_folder = Path(self.y_output)
        for i, features in enumerate(labelled_features):
            name_file = str(track_number) + "_" + str(i) + ".npy"
            with open(x_folder / name_file, "wb") as f_x:
                np.save(f_x, features.features)
            with open(y_folder / name_file, "wb") as f_y:
                np.save(f_y, features.labels.labels)

    def create_data_set(self):
        """
        Create the dataset and export it into np files.

        :param output_path:
        """
        i_file = 0
        for i, folder in enumerate(self.files_location):
            names = set([Path(file).stem for file in os.listdir(folder)])
            log.info("Pre-processing folder " + str(i + 1) + "/" + str(len(self.files_location)))

            for id_name, name in enumerate(names):

                mid_file = folder / f"{name}.mid"
                wav_file = folder / f"{name}.wav"
                midi_file = folder / f"{name}.midi"

                if wav_file.is_file() and mid_file.is_file():
                    log.info(
                        "Pre-processing file " + name + "(" + str(id_name +1) + "/" + str(len(names)) + ") in folder " + str(
                            i + 1))
                    labelled_features = self.pre_processing(wav_file, mid_file)
                    self.export_data(labelled_features, i_file)
                    i_file += 1

                if wav_file.is_file() and midi_file.is_file():
                    log.info(
                        "Pre-processing file " + name + "(" + str(id_name + 1) + "/" + str(
                            len(names)) + ") in folder " + str(
                            i + 1))
                    labelled_features = self.pre_processing(wav_file, mid_file)
                    self.export_data(labelled_features, i_file)
                    i_file += 1

