"""Own implementation of the MFCC Coeficients feature extraction method."""
import numpy as np
from scipy import signal
from scipy import fft
from scipy.fftpack import dct


def normalize_signal(signal):
    """
    Normalize the signal.

    :param signal:
    :return:
    """
    return signal / np.max(np.abs(signal))


def freq_to_mel(f):
    """
    Converts f herz into m mels

    :param f: in Hz
    :return: mels
    """
    return 2595 * np.log10(1 + f / 700)


def mel_to_freq(mel):
    """
    Converts mels to freq

    :param mel:
    :return: freq in Hz
    """
    return 700 * (10 ** (mel / 2595) - 1)


# TODO CHANGE NAME!!!!!!
def get_mel_filter_points(
    min_freq, max_freq, n_mel_filter, FFT_size, sampling_rate=44100
):
    """
    Creates the mel filter points.

    First of all convert the min and max freq of the signal into mel and create an
    array of n_mel_filter +2 (to create the triangles)mel points spaced linearly.
    Later on we convert this mel points into frequencies and finally associate every of
    this frequencies to their corresponding sample index in the FFT

    :param min_freq:
    :param max_freq:
    :param n_mel_filter:
    :param FFT_size:
    :param sampling_rate:
    :return: sample index points of the filter over the FFT window.
    """

    freq_min_mel = freq_to_mel(min_freq)
    freq_max_mel = freq_to_mel(max_freq)

    mel_ranges = np.linspace(freq_min_mel, freq_max_mel, n_mel_filter + 2)
    freq_ranges = mel_to_freq(mel_ranges)

    # TODO : CHECK THE SIZES OF THE FFT ARRAY!!!!!!!
    return np.floor((freq_ranges * (FFT_size)) / max_freq).astype(int)
    # return np.floor((freq_ranges * (FFT_size + 1)) / sampling_rate).astype(int)


def get_mel_filters(min_freq, max_freq, FFT_size, n_mel_filter=20, sampling_rate=44100):

    filter_points = get_mel_filter_points(
        min_freq, max_freq, n_mel_filter, FFT_size, sampling_rate
    )

    filters = np.zeros((n_mel_filter, FFT_size))

    # TODO CHANGE THIS ...
    for n in range(len(filter_points) - 2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(
            0, 1, filter_points[n + 1] - filter_points[n]
        )
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(
            1, 0, filter_points[n + 2] - filter_points[n + 1]
        )

    return filters, filter_points


def feature_extraction_frame(
    audio, n_mel_filter_bins, num_dct_coef, sampling_freq=44100
):

    # Add 'reflect' padding
    padding_samples = int(audio.shape[0] * 0.4)
    if ((padding_samples + audio.shape[0]) / 2) % 2 == 1:
        padding_samples += 1

    signal_padded = np.pad(audio, (padding_samples, padding_samples), mode="reflect")

    # Normalize signal [-1, 1]
    signal_normalized = normalize_signal(signal_padded)

    # Apply Hann window (eliminates discontinuities)
    window = signal.get_window("hann", signal_normalized.shape[0], fftbins=True)
    signal_win = window * signal_normalized

    # FFT computation and take only positive part of the spectrum
    signal_fft = fft.fft(signal_win, axis=0)[: int(signal_win.shape[0] / 2 + 1)]
    size_fft = signal_fft.shape[0]

    # Signal power
    signal_power = np.sqrt(np.abs(signal_fft))

    # Obtain the different triangle filters
    min_freq_signal = 0
    max_freq_signal = sampling_freq / 2
    filters, mel_freqs = get_mel_filters(
        min_freq_signal, max_freq_signal, size_fft, n_mel_filter_bins
    )

    # Normalization of the gain of every triangle filter
    # librosa implementation
    enorm = 2.0 / (mel_freqs[2 : n_mel_filter_bins + 2] - mel_freqs[:n_mel_filter_bins])
    filters *= enorm[:, np.newaxis]

    # Filter the signal using the tr
    filtered_signal = np.dot(filters, signal_power)

    # signal in dB
    signal_log = 10 * np.log10(filtered_signal)

    # apply the discrete cosine transform to obtain the cepstral coefficients
    # It turns out that the result of the filter bank is very correlated, which
    # is unacceptable for feeding the data to our algorithm.
    num_dct_deficients = 40
    mfcc = dct(filtered_signal)[:num_dct_coef]

    return mfcc
