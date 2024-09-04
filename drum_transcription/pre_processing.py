import numpy as np
from dataclasses import dataclass
from typing import List
from librosa import feature
import mido as md
from drum_transcription import GROOVE_PITCH_NAMES

class Labels:
    """Class for the Labels"""

    def __init__(self):
        """Constructor of the class"""
        """9 combined classes of drum elements"""
        self.n_elements = 9
        self.labels = np.zeros((self.n_elements, ), dtype=int)

    def element_at(self, position):
        """
        It add a 1 in the position indicating that this element was hit
        :param position: index starting from 0
        """
        self.labels[position] = 1

    def is_empty(self):
        """Check if the labels contains only0 s """
        if np.sum(self.labels) > 0:
            return False
        else:
            return True
@dataclass
class LabeleledFeatures:
    """It will store the features in an audio frame from t_0 to t_f and its respective labels"""

    t_initial: float  # seconds
    features: np.array
    labels: Labels


class MidiMessages():

    def __init__(self, midi_file):
        self._time = 0
        self.tempo = None
        self.on_notes, self.on_times = self._get_midi_queue(midi_file)

    def _get_midi_queue(self, midi_file):

        mid = md.MidiFile(midi_file)
        if mid.type != 0:
            print("MIDI FILE WITH OTHER TYPE WHICH IS NOT 0")

        tpb = mid.ticks_per_beat

        on_notes = []
        on_times = []
        for track in mid.tracks:
            for msg in track:
                self._time += msg.time
                if msg.is_meta and msg.type == 'set_tempo':
                    self.tempo = msg.tempo
                    self._time = msg.time

                elif msg.type == 'note_on':
                    note = msg.note
                    seq_time = md.tick2second(self._time, tpb, self.tempo)
                    on_notes.append(note)
                    on_times.append(seq_time)

        return np.array(on_notes), np.array(on_times)

    def get_query_midi(self, t_0, t_f):

        ids = np.argwhere((t_0< self.on_times) & (self.on_times<t_f))[:, 0]
        if ids.shape[0] != 0:
            return self.on_notes[ids]
        else:
            return []

