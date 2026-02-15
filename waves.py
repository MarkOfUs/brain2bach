import time
from pathlib import Path

import numpy as np
import serial
import h5py  # pip install h5py

PORT = "/dev/cu.usbmodem14101"   # <-- change this
BAUD = 115200

TARGET_HZ = 50.0                 # pacing (best-effort)
BASENAME = "capture"

VREF = 3.3
ADC_MAX = 65535.0
SCALE = VREF / ADC_MAX           # raw_count * SCALE = volts

OUT_DIR = Path(".")
MAT_PATH = OUT_DIR / f"{BASENAME}.mat"

def parse_three_numbers(line: str):
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create/overwrite a MATLAB v7.3-compatible .mat (HDF5) with an extendable dataset
    with h5py.File(MAT_PATH, "w") as f:
        dset = f.create_dataset(
            "data",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype="f8",
            chunks=(1024, 4),
            compression="gzip",
            compression_opts=4,
        )

        # lightweight metadata as attrs (MATLAB can read these from HDF5 too)
        f.attrs["columns"] = np.array([b"t_sec", b"ch0_V", b"ch1_V", b"ch2_V"])
        f.attrs["port"] = PORT.encode("utf-8")
        f.attrs["baud"] = BAUD
        f.attrs["target_hz"] = TARGET_HZ
        f.attrs["created_unix"] = time.time()
        f.attrs["units_t_sec"] = b"s"
        f.attrs["units_ch"] = b"V"
        f.attrs["adc_vref"] = VREF
        f.attrs["adc_max_count"] = ADC_MAX
        f.attrs["scale_V_per_count"] = SCALE

        print(f"Recording... Ctrl+C to stop. Appending to {MAT_PATH} (MAT v7.3 / HDF5)")

        t_start = time.perf_counter()
        period = 1.0 / TARGET_HZ
        next_deadline = t_start

        n = 0
        try:
            with serial.Serial(PORT, BAUD, timeout=1) as ser:
                ser.reset_input_buffer()

                while True:
                    now = time.perf_counter()

                    # best-effort pacing
                    if now < next_deadline:
                        time.sleep(next_deadline - now)
                    next_deadline += period

                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    raw = parse_three_numbers(line)
                    if raw is None:
                        continue

                    # raw counts -> volts
                    v0 = raw[0] * SCALE
                    v1 = raw[1] * SCALE
                    v2 = raw[2] * SCALE

                    t_rel = time.perf_counter() - t_start
                    row = np.array([[t_rel, v0, v1, v2]], dtype=np.float64)

                    # append one row to dataset
                    dset.resize((n + 1, 4))
                    dset[n, :] = row[0]
                    n += 1

                    # ensure it’s actually written out frequently
                    # (still “appending” — not rewriting)
                    if (n % 50) == 0:
                        f.flush()
                        hz = n / max(1e-9, (time.perf_counter() - t_start))
                        print(f"{n} samples (~{hz:.1f} Hz)")

        except KeyboardInterrupt:
            f.flush()
            print(f"\nStopped. Final samples: {n} -> {MAT_PATH}")

if __name__ == "__main__":
    main()
