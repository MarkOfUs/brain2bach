from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import scipy.io
import scipy.signal
import h5py
import librosa
import soundfile as sf

import torch
import torch.nn as nn


# ------------------------
# Defaults (mirrors your repo script)
# ------------------------
TARGET_EEG_HZ = 100.0
SR_AUDIO = 22050
N_FFT = 1024
HOP = 220
N_MELS = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Where you drop checkpoints inside the container
BILSTM_CKPT = os.getenv("BILSTM_CKPT", "/app/models/sub-17_run5_bilstm.pt")
ENV2MEL_CKPT = os.getenv("ENV2MEL_CKPT", "/app/models/sub-17_run5_env2mel.pt")
REFINER_CKPT = os.getenv("REFINER_CKPT", "/app/models/refiner_final.pt")


# ------------------------
# MAT loading (supports old .mat and v7.3 HDF5 .mat)
# ------------------------
def _try_loadmat_bytes(mat_bytes: bytes) -> Dict[str, Any]:
    # Try scipy first (works for MATLAB <= v7.2)
    try:
        return scipy.io.loadmat(io.BytesIO(mat_bytes))
    except NotImplementedError:
        # Likely MATLAB v7.3 (HDF5). Fall through to h5py.
        pass
    except Exception:
        # Fall through (we'll try h5py next)
        pass

    # Try h5py (MATLAB v7.3)
    with h5py.File(io.BytesIO(mat_bytes), "r") as f:
        out: Dict[str, Any] = {}
        for k in f.keys():
            out[k] = np.array(f[k])
        return out


def estimate_fs(t_sec: np.ndarray) -> float:
    t = np.asarray(t_sec).reshape(-1)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) < 5:
        raise RuntimeError("t_sec does not look like a valid time vector.")
    return 1.0 / float(np.median(dt))


def resample_channels(x: np.ndarray, fs_in: float, fs_out: float, target_len: int) -> np.ndarray:
    # x: (C, T)
    c, t_in = x.shape
    t_out = int(round(t_in * fs_out / fs_in))
    y = scipy.signal.resample(x, t_out, axis=1)

    if y.shape[1] >= target_len:
        y = y[:, :target_len]
    else:
        pad = target_len - y.shape[1]
        y = np.pad(y, ((0, 0), (0, pad)))

    return y.astype(np.float32)


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    analytic = scipy.signal.hilbert(x, axis=1)
    env = np.abs(analytic)
    return env.astype(np.float32)


def expand_3_to_32(ch3: np.ndarray) -> np.ndarray:
    # ch3: (3, T) -> (32, T) using the same synthetic expansion as your script
    T = ch3.shape[1]
    out = np.zeros((32, T), dtype=np.float32)

    left = ch3[0]
    right = ch3[1]
    mid = ch3[2]

    base = [
        left,
        right,
        mid,
        0.5 * (left + right),
        0.5 * (left + mid),
        0.5 * (right + mid),
        0.333 * (left + right + mid),
        (left - right),
        (mid - 0.5 * (left + right)),
    ]

    for i in range(32):
        b = base[i % len(base)]
        scale = 1.0 - 0.01 * (i % 10)
        out[i] = scale * b

    return out


def normalize_like_training(eeg_env: np.ndarray) -> np.ndarray:
    mu = eeg_env.mean(axis=1, keepdims=True)
    sd = eeg_env.std(axis=1, keepdims=True) + 1e-6
    return ((eeg_env - mu) / sd).astype(np.float32)


def invert_log_mel(mel_log: np.ndarray, griffin_iters: int) -> np.ndarray:
    # Your script treats model output as log-mel.
    mel = np.exp(mel_log)
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=SR_AUDIO,
        n_fft=N_FFT,
        hop_length=HOP,
        power=2.0,
        n_iter=int(griffin_iters),
    )
    return audio.astype(np.float32)


# ------------------------
# Model defs (copied from your three_channel_to_song.py)
# ------------------------
HIDDEN_SIZE = 64
NUM_LAYERS = 2


class BiLSTMDecoder(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(2 * HIDDEN_SIZE, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.proj(h)


class EnvToMelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 80, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, ch, 5, padding=2),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(cin, cout, 5, padding=2),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )
        self.r1 = ResBlock(cout)
        self.r2 = ResBlock(cout)
        self.down = nn.Conv1d(cout, cout, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.pre(x)
        x = self.r1(x)
        x = self.r2(x)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, cin, skip_ch, cout):
        super().__init__()
        self.up = nn.ConvTranspose1d(cin, cout, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv1d(cout + skip_ch, cout, 5, padding=2),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )
        self.r1 = ResBlock(cout)
        self.r2 = ResBlock(cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            diff = skip.shape[-1] - x.shape[-1]
            if diff > 0:
                x = nn.functional.pad(x, (0, diff))
            else:
                x = x[..., : skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.r1(x)
        x = self.r2(x)
        return x


class AudioRefiner(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv1d(1, base, 7, padding=3),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.d1 = DownBlock(base, base)
        self.d2 = DownBlock(base, base * 2)
        self.d3 = DownBlock(base * 2, base * 4)
        self.mid = nn.Sequential(
            ResBlock(base * 4),
            ResBlock(base * 4),
            ResBlock(base * 4),
        )
        self.u3 = UpBlock(base * 4, base * 4, base * 2)
        self.u2 = UpBlock(base * 2, base * 2, base)
        self.u1 = UpBlock(base, base, base)
        self.out = nn.Conv1d(base, 1, 1)

    def forward(self, x):
        x0 = x
        x = self.inp(x)
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x = self.mid(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return x0 + self.out(x)


@dataclass
class MusicModels:
    bilstm: BiLSTMDecoder
    env2mel: EnvToMelNet
    refiner: Optional[AudioRefiner]


def load_models() -> MusicModels:
    # Lazy-load on first request via handler
    if not os.path.exists(BILSTM_CKPT):
        raise FileNotFoundError(
            f"Missing BiLSTM checkpoint: {BILSTM_CKPT}\n"
            "Drop it into ./models/ and rebuild, or set BILSTM_CKPT env var."
        )

    if not os.path.exists(ENV2MEL_CKPT):
        raise FileNotFoundError(
            f"Missing Env2Mel checkpoint: {ENV2MEL_CKPT}\n"
            "Drop it into ./models/ and rebuild, or set ENV2MEL_CKPT env var."
        )

    bilstm = BiLSTMDecoder(input_size=32).to(DEVICE)
    bilstm.load_state_dict(torch.load(BILSTM_CKPT, map_location=DEVICE))
    bilstm.eval()

    env2mel = EnvToMelNet().to(DEVICE)
    env2mel.load_state_dict(torch.load(ENV2MEL_CKPT, map_location=DEVICE))
    env2mel.eval()

    refiner = None
    if os.path.exists(REFINER_CKPT):
        refiner = AudioRefiner().to(DEVICE)
        refiner.load_state_dict(torch.load(REFINER_CKPT, map_location=DEVICE))
        refiner.eval()

    return MusicModels(bilstm=bilstm, env2mel=env2mel, refiner=refiner)


def mat_bytes_to_wav_bytes(
    models: MusicModels,
    mat_bytes: bytes,
    *,
    target_t: int = 2101,
    griffin_iters: int = 32,
    use_refiner: bool = True,
    mat_key_data: str = "data",
    mat_key_time: str = "t_sec",
    channel_cols: List[int] = [1, 2, 3],
) -> Tuple[bytes, Dict[str, Any], Optional[bytes]]:
    mat = _try_loadmat_bytes(mat_bytes)

    if mat_key_data not in mat or mat_key_time not in mat:
        raise RuntimeError(
            f"capture.mat must contain variables: {mat_key_data} and {mat_key_time} "
            f"(got keys: {list(mat.keys())[:20]})"
        )

    data = np.asarray(mat[mat_key_data])
    t_sec = np.asarray(mat[mat_key_time]).reshape(-1)

    if data.ndim != 2:
        raise RuntimeError(f"Unexpected data ndim: {data.ndim}, shape={data.shape}")

    if max(channel_cols) >= data.shape[1]:
        raise RuntimeError(
            f"channel_cols {channel_cols} out of bounds for data with shape {data.shape}. "
            "If your 'data' is shaped (channels, time) instead, transpose or adjust mat_key_data."
        )

    # Expect data is (time, channels). Pull 3 channels.
    ch = [data[:, c].astype(np.float32) for c in channel_cols]
    x3 = np.stack(ch, axis=0)  # (3, T)

    fs_in = estimate_fs(t_sec)
    x3_rs = resample_channels(x3, fs_in, TARGET_EEG_HZ, int(target_t))
    x32 = expand_3_to_32(x3_rs)

    eeg_env = hilbert_envelope(x32)
    eeg_env = normalize_like_training(eeg_env)

    # BiLSTM expects (B, T, 32)
    x = torch.from_numpy(eeg_env.T).unsqueeze(0).float().to(DEVICE)
    with torch.inference_mode():
        pred_env = models.bilstm(x).squeeze(0).squeeze(-1).detach().cpu().numpy()

    # Normalize envelope prediction (matches your script)
    pred_env = (pred_env - pred_env.mean()) / (pred_env.std() + 1e-6)

    # Env2Mel expects (B, 1, T)
    x_env = torch.from_numpy(pred_env[None, None, :]).float().to(DEVICE)
    with torch.inference_mode():
        pred_mel = models.env2mel(x_env).squeeze(0).detach().cpu().numpy()

    audio_gen = invert_log_mel(pred_mel, griffin_iters=int(griffin_iters))
    audio_final = audio_gen

    if use_refiner and models.refiner is not None:
        wav = torch.from_numpy(audio_gen).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            wav_ref = models.refiner(wav).squeeze().detach().cpu().numpy()
        audio_final = wav_ref.astype(np.float32)

    # Encode WAV bytes
    gen_buf = io.BytesIO()
    sf.write(gen_buf, audio_gen, SR_AUDIO, format="WAV", subtype="PCM_16")
    wav_gen_bytes = gen_buf.getvalue()

    buf = io.BytesIO()
    sf.write(buf, audio_final, SR_AUDIO, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    meta = {
        "device": DEVICE,
        "sr": SR_AUDIO,
        "fs_in_est_hz": float(fs_in),
        "target_t": int(target_t),
        "griffin_iters": int(griffin_iters),
        "used_refiner": bool(use_refiner and models.refiner is not None),
        "duration_s": float(len(audio_final) / SR_AUDIO),
        "wav_bytes": int(len(wav_bytes)),
        "mat_key_data": mat_key_data,
        "mat_key_time": mat_key_time,
        "channel_cols": channel_cols,
    }

    return wav_bytes, meta, wav_gen_bytes
