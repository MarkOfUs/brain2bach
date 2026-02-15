import numpy as np

class EEGSignalGenerator:
    def __init__(self, fs=250):
        self.fs = fs
        self.t = 0

    def get_samples(self, n_samples=10):
        time = np.arange(self.t, self.t + n_samples) / self.fs
        self.t += n_samples

        ch1 = np.sin(2 * np.pi * 10 * time) + 0.2 * np.random.randn(n_samples)
        ch2 = np.sin(2 * np.pi * 12 * time) + 0.2 * np.random.randn(n_samples)
        ch3 = np.sin(2 * np.pi * 8 * time) + 0.2 * np.random.randn(n_samples)

        return ch1, ch2, ch3
