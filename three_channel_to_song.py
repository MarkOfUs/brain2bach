from pathlib import Path
import numpy as np
import scipy.io
import scipy.signal
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import config


SUBJECT = "sub-17"
RUN_ID_FOR_MODELS = "5"
CAPTURE_MAT = Path(r"Downloads\capture.mat")

TARGET_EEG_HZ = 100.0
TARGET_T = 2101
SR_AUDIO = 22050
N_FFT = 1024
HOP = 220
N_MELS = 80
N_GRIFFIN_ITERS = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


OUT_DIR = config.processed_subject_dir(SUBJECT) / "listen_capture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_DIR = config.processed_subject_dir(SUBJECT) / "pairs"
BILSTM_PATH = PAIR_DIR / f"{SUBJECT}_run{RUN_ID_FOR_MODELS}_bilstm.pt"
ENV2MEL_PATH = PAIR_DIR / f"{SUBJECT}_run{RUN_ID_FOR_MODELS}_env2mel.pt"


REFINER_PATH = Path(r"BachBrain\code\refiner_final.pt")
USE_REFINER = REFINER_PATH.exists()



def estimate_fs(t_sec: np.ndarray) -> float:
    t = np.asarray(t_sec).reshape(-1)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) < 5:
        raise RuntimeError("t_sec does not look like a valid time vector.")
    return 1.0 / np.median(dt)



def resample_channels(x: np.ndarray, fs_in: float, fs_out: float, target_len: int) -> np.ndarray:
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



def invert_log_mel(mel_log: np.ndarray) -> np.ndarray:
    mel = np.exp(mel_log)
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=SR_AUDIO,
        n_fft=N_FFT,
        hop_length=HOP,
        power=2.0,
        n_iter=N_GRIFFIN_ITERS,
    )
    return audio.astype(np.float32)



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
                x = x[..., :skip.shape[-1]]
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



def main():
    print("Loading capture file:")
    print(f"  {CAPTURE_MAT}")

    mat = scipy.io.loadmat(CAPTURE_MAT)
    if "data" not in mat or "t_sec" not in mat:
        raise RuntimeError("capture.mat must contain variables: data and t_sec")

    data = np.asarray(mat["data"])
    t_sec = np.asarray(mat["t_sec"]).reshape(-1)

    if data.ndim != 2 or data.shape[1] < 4:
        raise RuntimeError(f"Unexpected data shape: {data.shape}")

    ch0 = data[:, 1].astype(np.float32)
    ch1 = data[:, 2].astype(np.float32)
    ch2 = data[:, 3].astype(np.float32)

    x3 = np.stack([ch0, ch1, ch2], axis=0)

    fs_in = estimate_fs(t_sec)
    print(f"Estimated sampling rate from t_sec: {fs_in:.3f} Hz")

    x3_rs = resample_channels(x3, fs_in, TARGET_EEG_HZ, TARGET_T)
    x32 = expand_3_to_32(x3_rs)

    eeg_env = hilbert_envelope(x32)
    eeg_env = normalize_like_training(eeg_env)

    print(f"EEG envelope shape: {eeg_env.shape}")

    if not BILSTM_PATH.exists():
        raise FileNotFoundError(f"Missing BiLSTM model: {BILSTM_PATH}")

    bilstm = BiLSTMDecoder(input_size=32).to(DEVICE)
    bilstm.load_state_dict(torch.load(BILSTM_PATH, map_location=DEVICE))
    bilstm.eval()

    x = torch.from_numpy(eeg_env.T).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred_env = bilstm(x).squeeze(0).squeeze(-1).cpu().numpy()

    pred_env = (pred_env - pred_env.mean()) / (pred_env.std() + 1e-6)

    if not ENV2MEL_PATH.exists():
        raise FileNotFoundError(f"Missing Env2Mel model: {ENV2MEL_PATH}")

    env2mel = EnvToMelNet().to(DEVICE)
    env2mel.load_state_dict(torch.load(ENV2MEL_PATH, map_location=DEVICE))
    env2mel.eval()

    x_env = torch.from_numpy(pred_env[None, None, :]).float().to(DEVICE)
    with torch.no_grad():
        pred_mel = env2mel(x_env).squeeze(0).cpu().numpy()

    audio_gen = invert_log_mel(pred_mel)

    audio_final = audio_gen
    if USE_REFINER:
        ref = AudioRefiner().to(DEVICE)
        ref.load_state_dict(torch.load(REFINER_PATH, map_location=DEVICE))
        ref.eval()

        wav = torch.from_numpy(audio_gen).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            wav_ref = ref(wav).squeeze().cpu().numpy()
        audio_final = wav_ref.astype(np.float32)

    out_gen = OUT_DIR / "capture_generated.wav"
    out_final = OUT_DIR / "capture_final_refined.wav"

    sf.write(out_gen, audio_gen, SR_AUDIO)
    sf.write(out_final, audio_final, SR_AUDIO)

    print("\nSaved:")
    print(f"  Generated: {out_gen}")
    print(f"  Final refined: {out_final}")
    print(f"Folder: {OUT_DIR}")



if __name__ == "__main__":
    main()
