import torch
import torchaudio
import typing

DEFAULT_HOP_LENGTH_MULTIPLIER: typing.Final = 4


class ISTFT(object):
    def __init__(self, n_fft=400, hop_length=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is None else self.n_fft // DEFAULT_HOP_LENGTH_MULTIPLIER

    def __call__(self, sample):
        return torch.istft(sample, self.n_fft, self.hop_length)


class Spectrogram(object):
    def __init__(self, inverse=False, kind='real', **kwargs):
        super().__init__()
        if kind not in ['real', 'complex']:
            raise Exception(
                "Invalid spectrogram type. Must be `real` or `complex`")
        self.type = kind
        self.is_inverse = inverse
        self.log2 = 'log2' in kwargs and kwargs['log2']

        self.n_fft = kwargs['n_fft'] if 'n_fft' in kwargs else 400
        self.hop_length = kwargs[
            'hop_length'] if 'hop_length' in kwargs else self.n_fft // DEFAULT_HOP_LENGTH_MULTIPLIER

        if self.type in ['real', 'complex']:
            if inverse and self.type == 'real':
                self.transform = torchaudio.transforms.GriffinLim(
                    hop_length=self.hop_length)
            elif inverse and self.type == 'complex':
                self.transform = ISTFT(self.n_fft, self.hop_length)
            else:
                self.transform = torchaudio.transforms.Spectrogram(
                    power=2 if self.type == 'real' else None,
                    hop_length=self.hop_length)

    def inverse(self):
        """Return the inverse transform for the current settings"""
        return Spectrogram(inverse=not self.is_inverse,
                           type=self.type,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length)

    def __call__(self, sample):
        if self.is_inverse and self.log2:
            return torch.pow(2, self.transform(sample))
        elif self.log2:
            return torch.log2(self.transform(sample))
        else:
            return self.transform(sample)
