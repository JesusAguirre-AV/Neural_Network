from __future__ import annotations


"""
database.py
------------
Module responsible for building the training and testing feature databases.

It performs:
  - audio loading, resampling, and normalization
  - feature extraction using configurable boolean toggles
  - dataset construction (train/test)
  - saving .csv .xls and  .json artifacts for later model training
"""

import os, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
import matplotlib
matplotlib.use("Agg")  # so it can render without a GUI (safe on servers)
import matplotlib.pyplot as plt
import librosa.display

#Global constants controlling audio preprocessing

TARGET_SR = 22050  # Target sampling rate for all audio
MONO = True  # Convert to mono channel
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Step size between frames
WIN_LENGTH = None  # Window length; defaults to N_FFT if None


#Audio loading and normalization

def load_resample_normalize(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file, resample it to TARGET_SR, and normalize amplitude

    Parameters
    ----------
    path : str
        Path to the audio file

    Returns
    -------
    y : np.ndarray
         waveform samples normalized to [-1, 1].
    sr : int
         sampling rate after resampling.
    """
    y, sr = librosa.load(path, sr=None, mono=MONO)
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y.astype(np.float32), sr


#Feature selection and configuration

@dataclass
class FeatureConfig:
    """
    Boolean configuration specifying which audio features to include

    Each feature corresponds to a Librosa extractor. Setting a flag to True
    enables that feature when building the dataset.
    """
    mfcc: bool = True
    mfcc_delta: bool = False
    chroma: bool = True
    spectral_contrast: bool = True
    zcr: bool = True
    spectral_centroid: bool = False
    spectral_bandwidth: bool = False
    spectral_rolloff: bool = False
    rms: bool = False
    tempo: bool = False

    n_mfcc: int = 20
    aggregation: str = "mean_std"  # 'mean' | 'median' | 'mean_std'


#Feature extraction helpers

def _aggregate(feat: np.ndarray, how: str) -> np.ndarray:
    """
    Aggregate a time–frequency feature matrix into a fixed-length vector

    Parameters
    ----------
    feat : np.ndarray
        Feature matrix of shape (num_features, num_frames)
    how : str
        Aggregation method: 'mean', 'median', or 'mean_std'

    Returns
    -------
    np.ndarray
        Aggregated feature vector.
    """
    if feat.ndim == 1:
        feat = feat[None, :]
    if how == "mean":
        return np.mean(feat, axis=1)
    if how == "median":
        return np.median(feat, axis=1)
    if how == "mean_std":
        mu = np.mean(feat, axis=1)
        sd = np.std(feat, axis=1)
        return np.concatenate([mu, sd], axis=0)
    raise ValueError(f"Unknown aggregate mode: {how}")


def extract_features(y: np.ndarray, sr: int, cfg: FeatureConfig) -> Dict[
    str, np.ndarray]:
    """
    Extract all enabled features from a waveform

    Parameters
    ----------
    y : np.ndarray
        Waveform array
    sr : int
        Sampling rate
    cfg : FeatureConfig
        Feature selection and aggregation configuration

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping feature names -> aggregated feature vectors
    """
    n_fft = N_FFT
    hop = HOP_LENGTH
    win = WIN_LENGTH or n_fft
    feats: Dict[str, np.ndarray] = {}

    # MFCCs
    if cfg.mfcc:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc,
                                    n_fft=n_fft, hop_length=hop, win_length=win)
        feats["mfcc"] = _aggregate(mfcc, cfg.aggregation)
        if cfg.mfcc_delta:
            d1 = librosa.feature.delta(mfcc, order=1)
            d2 = librosa.feature.delta(mfcc, order=2)
            feats["mfcc_delta"] = _aggregate(d1, cfg.aggregation)
            feats["mfcc_delta2"] = _aggregate(d2, cfg.aggregation)

    #chroma
    if cfg.chroma:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                             n_fft=n_fft, hop_length=hop)
        feats["chroma"] = _aggregate(chroma, cfg.aggregation)

    #spectral contrast
    if cfg.spectral_contrast:
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        feats["spec_contrast"] = _aggregate(contrast, cfg.aggregation)

    #zero-crossing rate
    if cfg.zcr:
        z = librosa.feature.zero_crossing_rate(y, hop_length=hop).squeeze()
        feats["zcr"] = _aggregate(z, cfg.aggregation)

    #additional spectral features
    if cfg.spectral_centroid:
        c = librosa.feature.spectral_centroid(y=y, sr=sr).squeeze()
        feats["centroid"] = _aggregate(c, cfg.aggregation)

    if cfg.spectral_bandwidth:
        b = librosa.feature.spectral_bandwidth(y=y, sr=sr).squeeze()
        feats["bandwidth"] = _aggregate(b, cfg.aggregation)

    if cfg.spectral_rolloff:
        r = librosa.feature.spectral_rolloff(y=y, sr=sr).squeeze()
        feats["rolloff"] = _aggregate(r, cfg.aggregation)

    #rms energy
    if cfg.rms:
        p = librosa.feature.rms(y=y).squeeze()
        feats["rms"] = _aggregate(p, cfg.aggregation)

    #tempo
    if cfg.tempo:
        t, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = np.array([t], dtype=float)

    return feats


def concat_feature_dict(d: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple feature vectors (dict values) into a single long vector

    Parameters
    ----------
    d : Dict[str, np.ndarray]
        Mapping from feature name to aggregated feature vector

    Returns
    -------
    np.ndarray
        1-D concatenated feature vector (sorted by feature name)
    """
    keys = sorted(d.keys())
    return np.concatenate([d[k] for k in keys], axis=0)



#Dataset builders

#just wanted to make modular
AUDIO_EXTS = (".au", ".wav", ".mp3")


def _iter_audio_files(root: str):
    """
    Recursively yield all supported audio file paths under a directory

    Parameters
    ----------
    root : str
        Directory to scan

    Yields
    ------
    str
        Full path to each audio file
    """
    for dp, _, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith(AUDIO_EXTS):
                yield os.path.join(dp, f)


def build_train_dataframe(train_root: str, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Construct the training feature DataFrame

    Expects subfolders of `train_root` where each subfolder name is treated
    as the class label

    Parameters
    ----------
    train_root : str
        Path to the root directory containing class subfolders
    cfg : FeatureConfig
        Configuration controlling which features to extract

    Returns
    -------
    pd.DataFrame
        DataFrame with n columns, label, and path
    """
    rows = []
    for class_dir in sorted(
            [p for p in Path(train_root).glob("*") if p.is_dir()]):
        label = class_dir.name
        for fp in _iter_audio_files(str(class_dir)):
            try:
                y, sr = load_resample_normalize(fp)
                fd = extract_features(y, sr, cfg)
                vec = concat_feature_dict(fd)
                rows.append({**{f"f{i}": v for i, v in enumerate(vec)},
                             "label": label, "path": str(fp)})
            except Exception as e:
                print(f"[warn] {fp}: {e}")
    return pd.DataFrame(rows)


def build_test_dataframe(test_root: str, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Construct the test feature DataFrame

    If one or more .txt files exist under test_root, each line is interpreted
    as a relative or absolute path to a test audio file. Otherwise all
    audio files under test_root are scanned directly

    Parameters
    ----------
    test_root : str
        Root directory containing test audio or a .txt list of paths
    cfg : FeatureConfig
        Configuration controlling which features to extract

    Returns
    -------
    pd.DataFrame
        DataFrame with N columns and path
    """
    rows = []
    txts = list(Path(test_root).glob("*.txt"))
    if txts:
        files: List[str] = []
        for t in txts:
            with open(t, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    P = Path(p)
                    if not P.is_absolute():
                        P = Path(test_root) / p
                    if P.exists():
                        files.append(str(P))
                    else:
                        print(f"[warn] listed path not found: {P}")
    else:
        files = [str(p) for p in Path(test_root).glob("*")
                 if p.suffix.lower() in AUDIO_EXTS]

    for fp in files:
        try:
            y, sr = load_resample_normalize(fp)
            fd = extract_features(y, sr, cfg)
            vec = concat_feature_dict(fd)
            rows.append(
                {**{f"f{i}": v for i, v in enumerate(vec)}, "path": str(fp)})
        except Exception as e:
            print(f"[warn] {fp}: {e}")
    return pd.DataFrame(rows)


# Saving artifacts

def save_database_artifacts(df_train: pd.DataFrame,
                            df_test: pd.DataFrame,
                            out_dir: str) -> None:
    """
    Save the processed datasets

    Outputs:
      - train_features.csv
      - test_features.csv
      - label_map.json
      - database.xlsx

    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset (features & labels + paths)
    df_test : pd.DataFrame
        Testing dataset (features & paths)
    out_dir : str
        Directory to save outputs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_csv = out / "train_features.csv"
    test_csv = out / "test_features.csv"
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    #create label mapping JSON
    if "label" in df_train.columns:
        classes = sorted(df_train["label"].astype(str).unique().tolist())
        cls2id = {c: i for i, c in enumerate(classes)}
        id2cls = {i: c for c, i in cls2id.items()}
        with open(out / "label_map.json", "w", encoding="utf-8") as f:
            json.dump({"class_to_id": cls2id, "id_to_class": id2cls}, f,
                      indent=2)

    # .xlxs files were weird so threw down this try and catch
    try:
        with pd.ExcelWriter(out / "database.xlsx", engine="openpyxl") as xw:
            df_train.to_excel(xw, sheet_name="train", index=False)
            df_test.to_excel(xw, sheet_name="test", index=False)
        print(f"[ok] wrote Excel -> {out / 'database.xlsx'}")
    except Exception as e:
        print(f"[info] Skipping Excel export ({e}). CSVs are written.")

    print(f"ok wrote CSVs -> {train_csv}, {test_csv}")



# Spectrogram generation
def compute_log_spectrogram(
        y: np.ndarray,
        sr: int,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute a log spectrogram from a waveform

    returns
    -------
    S_db : np.ndarray
        spectrogram in decibels with shape
    """
    win = WIN_LENGTH or n_fft
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win)
    S_power = np.abs(S) ** 2
    S_db = librosa.power_to_db(S_power, ref=np.max)
    return S_db


def save_spectrogram_image(
        S_db: np.ndarray,
        out_path: str,
        sr: int,
        hop_length: int = HOP_LENGTH,
        y_axis: str = "log",
) -> None:
    """
    Save a spectrogram image to disk (no axes)

    Parameters
    ----------
    S_db : np.ndarray
        Log spectrogram
    out_path : str
        output image path data/processed/spectrograms/train/CLASS/file.png
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        # no time axis labels for training images
        x_axis=None,
        # log frequency axis
        y_axis=y_axis,
    )
    plt.axis("off")
    plt.tight_layout(pad=0.0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def generate_spectrograms_for_train(train_root: str, out_root: str) -> None:
    """
    Generate spectrogram PNGs for all training audio files

    Output structure:
        out_root/train/<CLASS>/<basename>.png
    """
    train_root = Path(train_root)
    out_root = Path(out_root)

    class_dirs = [p for p in train_root.glob("*") if p.is_dir()]
    print(f"[spec] Generating spectrograms for train, classes: "
          f"{[d.name for d in class_dirs]}")

    for class_dir in sorted(class_dirs):
        label = class_dir.name
        for fp in _iter_audio_files(str(class_dir)):
            fp_path = Path(fp)
            out_path = out_root / "train" / label / (fp_path.stem + ".png")

            # skip if already exists
            if out_path.exists():
                continue

            try:
                y, sr = load_resample_normalize(fp)
                S_db = compute_log_spectrogram(y, sr)
                save_spectrogram_image(S_db, out_path, sr)
            except Exception as e:
                print(f"[spec warn] {fp}: {e}")


def _gather_test_files(test_root: str) -> List[str]:
    """
    helper to gather test files
    """
    test_root_path = Path(test_root)
    txts = list(test_root_path.glob("*.txt"))
    if txts:
        files: List[str] = []
        for t in txts:
            with open(t, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if not p:
                        continue
                    P = Path(p)
                    if not P.is_absolute():
                        P = test_root_path / p
                    if P.exists():
                        files.append(str(P))
                    else:
                        print(f"[spec warn] listed test path not found: {P}")
    else:
        files = [str(p) for p in test_root_path.glob("*")
                 if p.suffix.lower() in AUDIO_EXTS]
    return files


def generate_spectrograms_for_test(test_root: str, out_root: str) -> None:
    """
    Generate spectrogram PNGs for all test audio files

    Output structure
        out_root/test/<basename>.png
    """
    test_root = Path(test_root)
    out_root = Path(out_root)

    files = _gather_test_files(str(test_root))
    print(f"[spec] Generating spectrograms for test: {len(files)} files")

    for fp in files:
        fp_path = Path(fp)
        out_path = out_root / "test" / (fp_path.stem + ".png")

        #skip if already exists
        if out_path.exists():
            continue

        try:
            y, sr = load_resample_normalize(fp)
            S_db = compute_log_spectrogram(y, sr)
            save_spectrogram_image(S_db, out_path, sr)
        except Exception as e:
            print(f"[spec warn] {fp}: {e}")
