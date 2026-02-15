import serial
import time
import threading
import numpy as np

class SerialEEGReader:
    def __init__(self, port, baud, fs, vref=3.3, adc_max=65535):
        self.port = port
        self.baud = baud
        self.fs = fs
        self.vref = vref
        self.adc_max = adc_max

        self.latest_samples = np.zeros(3, dtype=np.float32)
        self.lock = threading.Lock()
        self.running = False

    def raw_to_volts(self, raw: int) -> float:
        return raw * self.vref / self.adc_max

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()

    def stop(self):
        self.running = False

    def _read_loop(self):
        with serial.Serial(self.port, self.baud, timeout=1) as ser:
            time.sleep(2)

            next_time = time.time()
            period = 1.0 / self.fs

            while self.running:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    continue

                try:
                    raw0, raw1, raw2 = map(int, parts)
                except ValueError:
                    continue

                sample = np.array([
                    self.raw_to_volts(raw0),
                    self.raw_to_volts(raw1),
                    self.raw_to_volts(raw2)
                ], dtype=np.float32)

                with self.lock:
                    self.latest_samples = sample

                next_time += period
                sleep_time = next_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def get_latest(self):
        with self.lock:
            return self.latest_samples.copy()
