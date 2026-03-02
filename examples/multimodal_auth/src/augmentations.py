"""
Augmentation wrappers for robustness evaluation.

Video augmentations are applied to preprocessed face frames (torch.Tensor, shape (3, 112, 112), float32 in [-1, 1]).
Audio augmentations are applied to audio windows (torch.Tensor, shape (1, samples), float32).

Each augmentor takes an `aug_type` and `severity` (1-4, where 1 = mild, 4 = severe).
"""

import math
import numpy as np
import torch
import torchaudio.functional as AF
import albumentations as A
from torchvision.transforms import ColorJitter
from typing import Literal


# ---------------------------------------------------------------------------
# Video Augmentations
# ---------------------------------------------------------------------------

# Severity-indexed parameter tables (index 0 = severity 1, etc.)
_VIDEO_PARAMS = {
    "jpeg_compression": {
        "quality": [70, 50, 30, 10],  # lower = more severe
    },
    "gaussian_blur": {
        "kernel": [3, 5, 7, 9],
    },
    "gaussian_noise": {
        "std": [0.01, 0.05, 0.1, 0.2],
    },
    "downscale": {
        "scale": [0.75, 0.5, 0.25, 0.15],
    },
    "color_jitter": {
        "strength": [0.2, 0.4, 0.6, 0.8],
    },
    "occlusion": {
        "num_holes": [1, 2, 3, 4],
        "hole_size": [10, 15, 20, 30],  # pixels (on 112x112)
    },
}

VIDEO_AUG_TYPES = list(_VIDEO_PARAMS.keys())


class VideoAugmentor:
    """Apply a single augmentation type at a given severity to face frames.

    Input:  torch.Tensor  (3, 112, 112)  float32 in [-1, 1] (after ImageNet-style normalisation)
    Output: torch.Tensor  (3, 112, 112)  float32 in [-1, 1]

    Albumentations expects uint8 HWC numpy, so we convert back and forth.
    """

    def __init__(self, aug_type: str, severity: int):
        if aug_type not in _VIDEO_PARAMS:
            raise ValueError(f"Unknown video augmentation: {aug_type}. Choose from {VIDEO_AUG_TYPES}")
        if severity < 1 or severity > 4:
            raise ValueError(f"Severity must be 1-4, got {severity}")

        self.aug_type = aug_type
        self.severity = severity
        self._idx = severity - 1
        self._build_transform()

    def _build_transform(self):
        p = _VIDEO_PARAMS[self.aug_type]
        idx = self._idx

        if self.aug_type == "jpeg_compression":
            q = p["quality"][idx]
            self._albu = A.Compose([A.ImageCompression(quality_range=(q, q), p=1.0)])
            self._mode = "albu"

        elif self.aug_type == "gaussian_blur":
            k = p["kernel"][idx]
            self._albu = A.Compose([A.GaussianBlur(blur_limit=(k, k), p=1.0)])
            self._mode = "albu"

        elif self.aug_type == "gaussian_noise":
            s = p["std"][idx]
            self._albu = A.Compose([A.GaussNoise(std_range=(s, s), p=1.0)])
            self._mode = "albu"

        elif self.aug_type == "downscale":
            s = p["scale"][idx]
            self._albu = A.Compose([A.Downscale(scale_range=(s, s), p=1.0)])
            self._mode = "albu"

        elif self.aug_type == "color_jitter":
            s = p["strength"][idx]
            self._jitter = ColorJitter(brightness=s, contrast=s, saturation=s * 0.5, hue=s * 0.1)
            self._mode = "torch"

        elif self.aug_type == "occlusion":
            nh = p["num_holes"][idx]
            hs = p["hole_size"][idx]
            self._albu = A.Compose([
                A.CoarseDropout(
                    num_holes_range=(nh, nh),
                    hole_height_range=(hs, hs),
                    hole_width_range=(hs, hs),
                    fill=0.0,
                    p=1.0,
                )
            ])
            self._mode = "albu"

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """Augment a single frame (3, H, W) float32 in [-1, 1]."""
        if self._mode == "torch":
            # ColorJitter expects [0, 1] range
            x = (frame + 1.0) / 2.0  # [-1,1] -> [0,1]
            x = self._jitter(x)
            return x * 2.0 - 1.0  # [0,1] -> [-1,1]

        # albumentations path: convert to uint8 HWC numpy
        x = (frame + 1.0) / 2.0  # [-1,1] -> [0,1]
        x_np = (x.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)  # (H, W, 3)
        augmented = self._albu(image=x_np)["image"]  # (H, W, 3) uint8
        x_t = torch.from_numpy(augmented).float() / 255.0  # (H, W, 3) [0,1]
        x_t = x_t.permute(2, 0, 1)  # (3, H, W)
        return x_t * 2.0 - 1.0  # [0,1] -> [-1,1]

    def __repr__(self):
        return f"VideoAugmentor(aug_type={self.aug_type!r}, severity={self.severity})"


# ---------------------------------------------------------------------------
# Audio Augmentations
# ---------------------------------------------------------------------------

_AUDIO_PARAMS = {
    "additive_noise": {
        "snr_db": [20, 15, 10, 5],  # lower = more noise
    },
    "bandpass": {
        "low_hz":  [300, 500, 700, 1000],
        "high_hz": [4000, 3400, 3000, 2500],
    },
    "speed_perturb": {
        "rate": [0.95, 0.9, 1.1, 1.15],
    },
    "reverb": {
        "rt60": [0.3, 0.6, 1.0, 1.5],
    },
    "volume": {
        "gain_db": [-5, -10, 5, 10],
    },
}

AUDIO_AUG_TYPES = list(_AUDIO_PARAMS.keys())


def _generate_rir(rt60: float, sr: int = 16000) -> torch.Tensor:
    """Generate a simple synthetic Room Impulse Response via exponential decay."""
    n_samples = int(rt60 * sr)
    if n_samples < 1:
        return torch.ones(1)
    t = torch.arange(n_samples, dtype=torch.float32)
    decay = torch.exp(-6.908 * t / n_samples)  # -60 dB at n_samples
    # Add some early reflections
    rir = torch.randn(n_samples) * decay
    rir[0] = 1.0  # direct path
    rir = rir / rir.abs().max()
    return rir


class AudioAugmentor:
    """Apply a single augmentation type at a given severity to audio windows.

    Input:  torch.Tensor  (1, samples)  float32
    Output: torch.Tensor  (1, samples)  float32
    """

    def __init__(self, aug_type: str, severity: int, sample_rate: int = 16000):
        if aug_type not in _AUDIO_PARAMS:
            raise ValueError(f"Unknown audio augmentation: {aug_type}. Choose from {AUDIO_AUG_TYPES}")
        if severity < 1 or severity > 4:
            raise ValueError(f"Severity must be 1-4, got {severity}")

        self.aug_type = aug_type
        self.severity = severity
        self.sample_rate = sample_rate
        self._idx = severity - 1

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Augment audio window (1, samples) or (samples,)."""
        squeeze = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze = True

        out = self._apply(audio)

        if squeeze:
            out = out.squeeze(0)
        return out

    def _apply(self, audio: torch.Tensor) -> torch.Tensor:
        p = _AUDIO_PARAMS[self.aug_type]
        idx = self._idx
        original_len = audio.shape[-1]

        if self.aug_type == "additive_noise":
            snr_db = p["snr_db"][idx]
            signal_power = audio.pow(2).mean()
            noise = torch.randn_like(audio)
            noise_power = noise.pow(2).mean()
            scale = torch.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
            return audio + noise * scale

        elif self.aug_type == "bandpass":
            low = p["low_hz"][idx]
            high = p["high_hz"][idx]
            # Apply cascaded biquad filters
            out = AF.highpass_biquad(audio, self.sample_rate, low)
            out = AF.lowpass_biquad(out, self.sample_rate, high)
            return out

        elif self.aug_type == "speed_perturb":
            rate = p["rate"][idx]
            # Resample to simulate speed change, then resample back
            new_sr = int(self.sample_rate * rate)
            out = AF.resample(audio, self.sample_rate, new_sr)
            out = AF.resample(out, new_sr, self.sample_rate)
            # Pad or trim to original length
            if out.shape[-1] > original_len:
                out = out[..., :original_len]
            elif out.shape[-1] < original_len:
                pad = original_len - out.shape[-1]
                out = torch.nn.functional.pad(out, (0, pad))
            return out

        elif self.aug_type == "reverb":
            rt60 = p["rt60"][idx]
            rir = _generate_rir(rt60, self.sample_rate)
            # Convolve: pad audio, apply conv, trim
            rir = rir.unsqueeze(0).unsqueeze(0)  # (1, 1, L_rir)
            audio_padded = torch.nn.functional.pad(audio.unsqueeze(0), (rir.shape[-1] - 1, 0))
            out = torch.nn.functional.conv1d(audio_padded, rir).squeeze(0)
            out = out[..., :original_len]
            # Normalise to prevent clipping
            peak = out.abs().max()
            if peak > 0:
                out = out * (audio.abs().max() / peak)
            return out

        elif self.aug_type == "volume":
            gain_db = p["gain_db"][idx]
            return audio * (10 ** (gain_db / 20))

    def __repr__(self):
        return f"AudioAugmentor(aug_type={self.aug_type!r}, severity={self.severity}, sr={self.sample_rate})"


# ---------------------------------------------------------------------------
# Combined / utility
# ---------------------------------------------------------------------------

ALL_VIDEO_AUGS = VIDEO_AUG_TYPES
ALL_AUDIO_AUGS = AUDIO_AUG_TYPES
ALL_SEVERITIES = [1, 2, 3, 4]


def get_augmentor(
    modality: Literal["video", "audio"],
    aug_type: str,
    severity: int,
    sample_rate: int = 16000,
):
    """Factory function for creating augmentors."""
    if modality == "video":
        return VideoAugmentor(aug_type, severity)
    elif modality == "audio":
        return AudioAugmentor(aug_type, severity, sample_rate)
    else:
        raise ValueError(f"Unknown modality: {modality}")
