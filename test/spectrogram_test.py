import unittest
import torch
import math
from spectrogram import Spectrogram
from typing import Final

COMPLEX: Final = 'complex'
REAL: Final = 'real'


class SpectrogramTest(unittest.TestCase):

    def test_complex(self):
        wave = torch.sin(torch.linspace(0, math.pi * 1000, steps=16000))
        for n in [100, 200, 400, 800]:
            for log in [False, True]:
                sc = Spectrogram(inverse=False, space=COMPLEX, n_fft=n, log2=log)
                isc = sc.inverse()
                isc_m = Spectrogram(inverse=True, space=COMPLEX, n_fft=n, log2=log)
                iisc = isc.inverse()

                self.assertTrue(torch.all(torch.isclose(isc_m(sc(wave)), isc(sc(wave)))))
                self.assertTrue(torch.all(torch.isclose(isc_m(sc(wave)), isc_m(iisc(wave)))))
                self.assertTrue(sc(wave).dtype in [torch.complex64, torch.complex128])

    def test_real(self):
        wave = torch.sin(torch.linspace(0, math.pi * 1000, steps=16000))
        for log in [False, True]:
            sc = Spectrogram(inverse=False, space=REAL, log2=log)
            isc = sc.inverse()
            isc_m = Spectrogram(inverse=True, space=REAL, log2=log)
            iisc = isc.inverse()

            torch.manual_seed(0)
            a = sc(wave)
            torch.manual_seed(0)
            b = iisc(wave)
            self.assertTrue(torch.all(torch.isclose(a, b)))
            self.assertEqual(a.size(), torch.Size([201, 161]))
            self.assertEqual(isc_m(a).size(), torch.Size([16000]))


if __name__ == '__main__':
    unittest.main()
