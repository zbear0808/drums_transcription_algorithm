import os.path

from drum_transcription.predict import convert_midi
import pytest
import numpy as np
import mido as md
import pathlib as Path


def test_convert_midi():

    # Tested with 10 notes
    notes = [36, 38, 40, 37, 48, 50, 45, 47, 43, 58]  # , 46, 26, 42, 22, 44, 49, 55, 57, 52, 51, 59, 53]
    time_note = []
    tempo = md.bpm2tempo(120)
    delta_t = 1  # sec

    midi_path = str(Path.Path(__file__).parent / "test_files" / "test_write_midi")

    for i, note in enumerate(notes):
        if i == 2:
            time_note.append([delta_t * (i - 1) + 0.075, [note]])
        elif i == 5:
            time_note.append([delta_t * (i - 1) + 0.005, [note]])
        elif i == 6:
            time_note.append([time_note[i - 1][0] + 0.025, [note]])
        else:
            time_note.append([delta_t * i, [note]])

    midi = convert_midi(midi_path, time_note, 120)
    tpb = midi.ticks_per_beat

    midi_path += '.mid'
    assert os.path.isfile(midi_path)

    time = 0
    cnt = 0
    precision = 0.001
    for track in midi.tracks:
        for i, msg in enumerate(track):
            time += msg.time
            if msg.type == "note_on":
                if cnt == 0:
                    assert time_note[cnt][0] == md.tick2second(time, tpb, tempo)
                else:
                    assert time_note[cnt][0] * (1 - precision) <= md.tick2second(time, tpb, tempo) < time_note[cnt][0] * (1 + precision)
                assert time_note[cnt][1][0] == msg.note
                cnt += 1