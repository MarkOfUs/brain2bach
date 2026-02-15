\
"""
EEG emotion inference from a MATLAB .mat payload.

This file is adapted from the user's EmotionInference.ipynb:
- Differential entropy features per band & channel
- FACED channel grid transform (via torcheeg)
- Small CNN classifier
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, lfilter

import torch
import torch.nn as nn

# torcheeg provides FACED_CHANNEL_LOCATION_DICT and the ToGrid/ToTensor transforms.
from torcheeg import transforms
from torcheeg.datasets.constants import FACED_CHANNEL_LOCATION_DICT


# -------- Defaults (override via handler input if you want) --------
DEFAULT_WINDOW_START = int(os.getenv("WINDOW_START", "0"))
DEFAULT_WINDOW_END   = int(os.getenv("WINDOW_END", "250"))
FALLBACK_FS          = float(os.getenv("FALLBACK_FS", "250.0"))

DEFAULT_MAT_CHANNEL_NAMES = os.getenv("MAT_CHANNEL_NAMES", "FP1,FZ,FP2").split(",")

EMO_FALLBACK = [
    "anger","disgust","fear","sadness","neutral","amusement","inspiration","joy","tenderness"
]

BANDS: List[Tuple[str, float, Optional[float]]] = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, None),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    """Tiny CNN that expects input shape (N, 5, H, W)."""
    def __init__(self, in_channels: int, num_classes: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _safe_filter(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Prefer filtfilt but fall back if the signal is too short."""
    try:
        return filtfilt(b, a, x, axis=-1)
    except ValueError:
        return lfilter(b, a, x, axis=-1)


def butter_filter_1d(x: np.ndarray, fs: float, low: Optional[float] = None, high: Optional[float] = None, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if nyq <= 0:
        raise ValueError(f"Bad fs={fs}")

    if low is not None and low >= nyq:
        return np.zeros_like(x)
    if high is not None and high >= nyq:
        high = nyq - 1e-3
        if high <= 0:
            return np.zeros_like(x)

    if low is None and high is None:
        return x

    if low is None:
        wn = high / nyq
        b, a = butter(order, wn, btype="lowpass")
    elif high is None:
        wn = low / nyq
        b, a = butter(order, wn, btype="highpass")
    else:
        if low >= high:
            return np.zeros_like(x)
        wn = [low / nyq, high / nyq]
        b, a = butter(order, wn, btype="bandpass")

    return _safe_filter(x, b, a)


def compute_de_BxC(eeg_CxT: np.ndarray, fs: float, eps: float = 1e-8) -> np.ndarray:
    """
    Differential entropy features:
      input:  C x T
      output: B x C  (B=5 frequency bands)
    """
    C, _ = eeg_CxT.shape
    F = np.zeros((len(BANDS), C), dtype=np.float32)

    for bi, (_, lo, hi) in enumerate(BANDS):
        for ci in range(C):
            x = eeg_CxT[ci].astype(np.float64)
            xf = butter_filter_1d(x, fs, low=lo, high=hi, order=4)
            var = np.var(xf) + eps
            de = 0.5 * np.log(2.0 * np.pi * np.e * var)
            F[bi, ci] = np.float32(de)

    return F


def _load_mat_from_bytes(mat_bytes: bytes) -> Dict[str, Any]:
    """
    Tries scipy.io.loadmat first. If it's a v7.3 (HDF5) .mat, fall back to h5py.
    """
    try:
        return sio.loadmat(io.BytesIO(mat_bytes))
    except NotImplementedError:
        import h5py  # lazy import
        out: Dict[str, Any] = {}
        with h5py.File(io.BytesIO(mat_bytes), "r") as f:
            def visit(name, obj):
                # store datasets as numpy arrays
                if isinstance(obj, h5py.Dataset):
                    out[name] = np.array(obj)
            f.visititems(visit)
        return out
    except Exception:
        # Re-raise with a clearer message
        raise


def load_eeg_from_mat_bytes(mat_bytes: bytes, fallback_fs: float = FALLBACK_FS) -> Tuple[np.ndarray, float]:
    """
    Returns:
      eeg_CxT (float64), fs (float)
    Looks for a few common layouts (including the one used in the notebook).
    """
    mat = _load_mat_from_bytes(mat_bytes)

    fs = None
    if "meta" in mat:
        try:
            meta = mat["meta"][0, 0]
            if hasattr(meta, "dtype") and meta.dtype.names:
                if "achieved_hz" in meta.dtype.names:
                    fs = float(meta["achieved_hz"].item())
                elif "target_hz" in meta.dtype.names:
                    fs = float(meta["target_hz"].item())
        except Exception:
            fs = None
    if fs is None:
        fs = float(fallback_fs)

    # Notebook format: mat["data"] is 2D, first column is timestamp, remaining are channels.
    if "data" in mat and isinstance(mat["data"], np.ndarray) and mat["data"].ndim == 2:
        arr = mat["data"]
        if arr.shape[1] >= 2:
            eeg_TxC = arr[:, 1:]
            return eeg_TxC.T.astype(np.float64), fs

    # Common key variants
    for key in ["eeg", "EEG", "X", "signals", "data_eeg"]:
        if key in mat and isinstance(mat[key], np.ndarray):
            x = np.squeeze(mat[key])
            if x.ndim == 2:
                # Decide whether it's CxT or TxC
                if x.shape[0] <= x.shape[1]:
                    return x.astype(np.float64), fs
                return x.T.astype(np.float64), fs

    raise KeyError(f"Couldn't find EEG array in .mat. Keys: {list(mat.keys())}")


def faced_channel_order() -> List[str]:
    return list(FACED_CHANNEL_LOCATION_DICT.keys())


def build_full_faced_feature(F_BxC_small: np.ndarray, mat_channel_names: Sequence[str]) -> np.ndarray:
    """
    Expand from BxC_small (e.g. 5x3) to full FACED channel list (5x30),
    placing the provided channels at their FACED positions and leaving others at 0.
    """
    faced_ch = faced_channel_order()
    F_full = np.zeros((len(BANDS), len(faced_ch)), dtype=np.float32)

    name_to_idx = {name: i for i, name in enumerate(faced_ch)}
    for j, name in enumerate(mat_channel_names):
        if name not in name_to_idx:
            raise ValueError(f"Channel name '{name}' not in FACED_CHANNEL_LOCATION_DICT keys.")
        F_full[:, name_to_idx[name]] = F_BxC_small[:, j]

    return F_full


def togrid_like_training(F_full_BxC: np.ndarray) -> np.ndarray:
    to_grid = transforms.ToGrid(FACED_CHANNEL_LOCATION_DICT)
    out = to_grid(eeg=F_full_BxC.T)  # (B,C)->(C,B) expected by ToGrid
    return out["eeg"] if isinstance(out, dict) else out


def totensor_like_training(G_BxHxW: np.ndarray) -> torch.Tensor:
    to_tensor = transforms.ToTensor()
    out = to_tensor(eeg=G_BxHxW)
    return out["eeg"] if isinstance(out, dict) else out


def normalize_like_training(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=(2, 3), keepdim=True)
    sd = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (x - mu) / sd


@dataclass
class InferenceResult:
    fs_hz: float
    window_start: int
    window_end: int
    mat_channels_used: List[str]
    probs: List[float]
    classes: List[str]
    topk: List[Dict[str, Any]]


def load_model(ckpt_path: str) -> Tuple[nn.Module, List[str]]:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    classes = EMO_FALLBACK
    if isinstance(ckpt, dict) and "meta" in ckpt and isinstance(ckpt["meta"], dict):
        if ckpt["meta"].get("classes") is not None:
            classes = list(ckpt["meta"]["classes"])

    model = SmallCNN(in_channels=len(BANDS), num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, classes


def predict_from_mat_bytes(
    mat_bytes: bytes,
    model: nn.Module,
    classes: Sequence[str],
    *,
    window_start: int = DEFAULT_WINDOW_START,
    window_end: int = DEFAULT_WINDOW_END,
    mat_channel_names: Optional[Sequence[str]] = None,
    topk: int = 3,
) -> InferenceResult:
    eeg_CxT, fs = load_eeg_from_mat_bytes(mat_bytes, fallback_fs=FALLBACK_FS)

    if eeg_CxT.shape[0] < 3:
        raise ValueError(f"Need at least 3 channels in mat, got {eeg_CxT.shape[0]}")

    # Use first 3 channels by default (matching the notebook).
    eeg3 = eeg_CxT[:3, :]

    eeg_win = eeg3[:, window_start:window_end]
    if eeg_win.shape[1] < 16:
        raise ValueError(f"Window too short after slicing: {eeg_win.shape}")

    F_small = compute_de_BxC(eeg_win, fs)

    names = list(mat_channel_names) if mat_channel_names is not None else list(DEFAULT_MAT_CHANNEL_NAMES)
    if len(names) != 3:
        raise ValueError(f"mat_channel_names must have length 3 (for 3 channels). Got {len(names)}")

    F_full = build_full_faced_feature(F_small, names)
    G = togrid_like_training(F_full)

    t = totensor_like_training(G)
    x = t.float().unsqueeze(0).to(DEVICE)
    x = normalize_like_training(x)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0].astype(float)

    k = int(max(1, min(topk, len(probs))))
    top_idx = probs.argsort()[-k:][::-1]

    top_list = [{"label": str(classes[i]), "prob": float(probs[i]), "rank": int(r + 1)} for r, i in enumerate(top_idx)]

    return InferenceResult(
        fs_hz=float(fs),
        window_start=int(window_start),
        window_end=int(window_end),
        mat_channels_used=names,
        probs=[float(p) for p in probs],
        classes=[str(c) for c in classes],
        topk=top_list,
    )
