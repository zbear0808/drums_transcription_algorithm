"""Tests for the mfcc_feature_extraction module."""
from drum_transcription.mfcc_feature_extraction import normalize_signal
from drum_transcription.mfcc_feature_extraction import freq_to_mel
from drum_transcription.mfcc_feature_extraction import mel_to_freq
from drum_transcription.mfcc_feature_extraction import feature_extraction_frame
import numpy as np


def test_freq_to_mel():

    input = 440  # hz
    expected_output = 549.6386754

    output = freq_to_mel(input)
    assert expected_output == np.round(output, 7)


def test_mel_to_freq():

    input = 1750
    expected_output = 2607.286634

    output = mel_to_freq(input)
    assert expected_output == np.round(output, 6)


def test_feature_extraction_frame():

    fs = 441000
    t = np.linspace(0, 0.01, int(0.01 * fs))
    toy_data = np.sin(np.pi * 1 * t)
    n_mfcc_coef = 10

    features_frame = feature_extraction_frame(toy_data, 10, 40, sampling_freq=44100)

    assert features_frame.shape[0] == 10


def test_normalize_signal():
    input = np.array([10, 2, 6, 8, 0, 1])
    expected_output = np.array([1, 0.2, 0.6, 0.8, 0, 0.1])

    output = normalize_signal(input)
    assert output.all() == expected_output.all()
