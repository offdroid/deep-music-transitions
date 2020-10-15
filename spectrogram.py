import torch
from torchaudio.transforms import GriffinLim
from typing import Final
from enum import Enum

DEFAULT_HOP_LENGTH_MULTIPLIER: Final = 4


class SpectrogramType(Enum):
    REAL = 1
    COMPLEX = 2


class Spectrogram(object):
    def __init__(self, inverse=False, space='real', log2=True, **kwargs):
        super().__init__()
        self.type = space if isinstance(space, SpectrogramType) else SpectrogramType[space.upper()]
        self.is_inverse = inverse
        self.log2 = log2

        self.n_fft = kwargs['n_fft'] if 'n_fft' in kwargs else 400
        self.hop_length = kwargs[
            'hop_length'] if 'hop_length' in kwargs else self.n_fft // DEFAULT_HOP_LENGTH_MULTIPLIER
        self.length = kwargs['length'] if 'length' in kwargs else None

        if self.real and self.is_inverse:
            self.griffinlim = GriffinLim(n_fft=self.n_fft, hop_length=self.hop_length, length=self.length)

    def __call__(self, x):
        return self._backward(x) if self.is_inverse else self._forward(x)

    @property
    def real(self) -> bool:
        return self.type == SpectrogramType.REAL

    @property
    def complex(self) -> bool:
        return self.type == SpectrogramType.COMPLEX

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length)
        x = torch.norm(x, 2, dim=-1) if self.real else torch.view_as_complex(x)
        return torch.log2(x) if self.log2 else x

    def _backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log2:
            x = torch.pow(2, x)
        return torch.istft(torch.view_as_real(x), n_fft=self.n_fft, hop_length=self.hop_length,
                           length=self.length) if self.complex else self.griffinlim(x)

    def inverse(self):
        """Return the inverse transform for the current settings"""
        return Spectrogram(inverse=not self.is_inverse,
                           space=self.type,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           log2=self.log2,
                           length=self.length)
