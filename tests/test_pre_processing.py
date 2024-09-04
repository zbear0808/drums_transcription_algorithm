from drum_transcription.pre_processing import MidiMessages

import pytest
import numpy as np
import pathlib as Path


def test_get_query_midi():

    path_midi = Path.Path(__file__).parent / "test_files" / "test_midi.mid"

    midi_msg = MidiMessages(path_midi)

    delta_t = 0.2  # seconds
    notes = [36, 38, 40, 37, 48, 50, 45, 47, 43, 58, 46, 26, 42, 22, 44, 49, 55, 57, 52, 51, 59, 53]
    times = np.arange(len(notes))*delta_t

    precision = 0.002
    for i, note in enumerate(notes):
        assert note == midi_msg.on_notes[i]
        if i == 0:
            assert times[i] == midi_msg.on_times[i]
        else:
            assert times[i]*(1-precision) <= midi_msg.on_times[i] < times[i]*(1+precision)


def test_pre_procceser():
    pytest.xfail("to be implemented")