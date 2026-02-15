import threading
import time
import queue
from pathlib import Path
import shutil

import numpy as np
from scipy.io import savemat

try:
    import serial
except ImportError:
    serial = None


class SerialEEGRecorder:
    """
    Reads 3-channel EEG from serial, streams latest samples,
    and periodically saves MAT v5 snapshots (RunPod compatible).
    """

    def __init__(
        self,
        port: str,
        baud: int,
        fs: int,
        snapshot_path: Path | None = None,
        snapshot_interval_s: float = 2.0,
    ):
        self.port = port
        self.baud = baud
        self.fs = fs

        self.snapshot_path = snapshot_path
        self.snapshot_interval_s = snapshot_interval_s

        self._stop_event = threading.Event()
        self._thread = None

        self._latest_sample = np.zeros(3, dtype=np.float64)
        self._latest_lock = threading.Lock()

        self._recording = False
        self._record_path = None

        self._rows = []
        self._t0 = None
        self._last_snapshot_time = 0.0

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[EEG] Recorder started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def get_latest(self):
        with self._latest_lock:
            return self._latest_sample.copy()

    def start_recording(self, path: Path):
        self._recording = True
        self._record_path = path
        self._rows.clear()
        self._t0 = time.perf_counter()
        print(f"[EEG] Recording started -> {path}")

    def stop_recording(self):
        self._recording = False
        if self._record_path:
            self._write_mat(self._record_path, final=True)
            print(f"[EEG] Recording stopped -> {self._record_path}")
        self._record_path = None

    # --------------------------------------------------
    # INTERNALS
    # --------------------------------------------------

    def _read_loop(self):
        if serial is None:
            print("[EEG] pyserial not installed, cannot read from serial port")
            return
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                ser.reset_input_buffer()
                period = 1.0 / self.fs

                while not self._stop_event.is_set():
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    sample = self._parse_line(line)
                    if sample is None:
                        continue

                    with self._latest_lock:
                        self._latest_sample[:] = sample

                    now = time.perf_counter()

                    if self._recording:
                        t_rel = now - self._t0
                        self._rows.append([t_rel, *sample])

                    if self.snapshot_path:
                        if now - self._last_snapshot_time >= self.snapshot_interval_s:
                            self._write_snapshot()
                            self._last_snapshot_time = now

                    time.sleep(period)

        except Exception as e:
            print("[EEG] Serial error:", e)

    def _parse_line(self, line: str):
        try:
            parts = line.split(",")
            if len(parts) != 3:
                return None
            return np.array(
                [float(parts[0]), float(parts[1]), float(parts[2])],
                dtype=np.float64,
            )
        except ValueError:
            return None

    # --------------------------------------------------
    # MAT FILE HANDLING
    # --------------------------------------------------

    def _write_snapshot(self):
        if not self._rows:
            return

        tmp = self.snapshot_path.with_suffix(".tmp.mat")
        self._write_mat(tmp, final=False)

        try:
            shutil.move(tmp, self.snapshot_path)
            print(
                f"[EEG] snapshot saved (MAT v5, {len(self._rows)} samples)"
            )
        except Exception as e:
            print("[EEG] snapshot replace failed:", e)

    def _write_mat(self, path: Path, final: bool):
        arr = np.asarray(self._rows, dtype=np.float64)
        if arr.size == 0:
            return

        data = {
            "data": arr,
            "t_sec": arr[:, 0],
            "meta": {
                "fs": self.fs,
                "num_samples": arr.shape[0],
                "final": bool(final),
                "created_unix": time.time(),
            },
        }

        savemat(path, data, do_compression=False)


class MockEEGRecorder:
    """
    Generates synthetic 3-channel EEG for demo/testing when no serial device.
    Same interface as SerialEEGRecorder.
    """

    def __init__(
        self,
        fs: int = 50,
        snapshot_path: Path | None = None,
        snapshot_interval_s: float = 2.0,
    ):
        self.fs = fs
        self.snapshot_path = snapshot_path
        self.snapshot_interval_s = snapshot_interval_s

        self._stop_event = threading.Event()
        self._thread = None

        self._latest_sample = np.zeros(3, dtype=np.float64)
        self._latest_lock = threading.Lock()

        self._recording = False
        self._record_path = None
        self._rows = []
        self._t0 = None
        self._last_snapshot_time = 0.0
        self._sample_counter = 0

    def start(self):
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._mock_loop, daemon=True)
        self._thread.start()
        print("[EEG] Mock recorder started (synthetic data)")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None

    def get_latest(self):
        with self._latest_lock:
            return self._latest_sample.copy()

    def start_recording(self, path: Path):
        self._recording = True
        self._record_path = Path(path) if not isinstance(path, Path) else path
        self._rows.clear()
        self._t0 = time.perf_counter()
        print(f"[EEG] Mock recording started -> {self._record_path}")

    def stop_recording(self):
        self._recording = False
        if self._record_path:
            self._write_mat(self._record_path, final=True)
            print(f"[EEG] Mock recording stopped -> {self._record_path}")
        self._record_path = None

    def _mock_loop(self):
        period = 1.0 / self.fs
        rng = np.random.default_rng(42)

        while not self._stop_event.is_set():
            t = self._sample_counter * period
            self._sample_counter += 1

            # Synthetic EEG-like signal (band-limited noise + slow waves)
            sample = (
                0.5 * np.sin(2 * np.pi * 2 * t)
                + 0.3 * np.sin(2 * np.pi * 8 * t)
                + 0.2 * rng.standard_normal(3)
            ).astype(np.float64)

            with self._latest_lock:
                self._latest_sample[:] = sample

            now = time.perf_counter()

            if self._recording:
                t_rel = now - self._t0
                self._rows.append([t_rel, float(sample[0]), float(sample[1]), float(sample[2])])

            if self.snapshot_path and self._recording and self._rows:
                if now - self._last_snapshot_time >= self.snapshot_interval_s:
                    self._write_snapshot()
                    self._last_snapshot_time = now

            time.sleep(period)

    def _write_snapshot(self):
        if not self._rows:
            return
        tmp = self.snapshot_path.with_suffix(".tmp.mat")
        self._write_mat(tmp, final=False)
        try:
            shutil.move(tmp, self.snapshot_path)
            print(f"[EEG] mock snapshot saved (MAT v5, {len(self._rows)} samples)")
        except Exception as e:
            print("[EEG] snapshot replace failed:", e)

    def _write_mat(self, path: Path, final: bool):
        arr = np.asarray(self._rows, dtype=np.float64)
        if arr.size == 0:
            return
        data = {
            "data": arr,
            "t_sec": arr[:, 0],
            "meta": {
                "fs": self.fs,
                "num_samples": arr.shape[0],
                "final": bool(final),
                "created_unix": time.time(),
            },
        }
        savemat(path, data, do_compression=False)
