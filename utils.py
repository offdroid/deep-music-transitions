import torch
import numpy as np
import sounddevice as sd


def play_audio(waveform, sample_rate, blocking=True):
    waveform = waveform.cpu().numpy().squeeze(axis=0)
    sd.play(waveform, samplerate=sample_rate, blocking=blocking)
