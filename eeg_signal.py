import numpy as np

class EEGGenerator:
    """
    Simple EEG signal simulator with three channels.

    This simulates EEG-like oscillations with noise.
    Sampling frequency is configurable.
    """

    def __init__(self, fs=250):
        self.fs = fs
        self.sample_index = 0

    def generate(self, n_samples=10):
        """
        Generate n_samples for three EEG channels.

        Returns:
            ch1, ch2, ch3 as Python lists (JSON serializable)
        """

        t = (self.sample_index + np.arange(n_samples)) / self.fs
        self.sample_index += n_samples

        # EEG-like rhythms (alpha-ish bands) + noise
        ch1 = 1.0 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(n_samples)
        ch2 = 0.8 * np.sin(2 * np.pi * 12 * t) + 0.2 * np.random.randn(n_samples)
        ch3 = 0.6 * np.sin(2 * np.pi * 8 * t)  + 0.2 * np.random.randn(n_samples)

        return ch1.tolist(), ch2.tolist(), ch3.tolist()
