import torch
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt


def play_audio(waveform, sample_rate, blocking=True):
    waveform = waveform.cpu().numpy().squeeze(axis=0)
    sd.play(waveform, samplerate=sample_rate, blocking=blocking)


def show_spectrogram(spectrogram):
    plt.figure()
    plt.imshow(spectrogram.cpu().numpy(), cmap='plasma')
    plt.gca().invert_yaxis()
    plt.show()
