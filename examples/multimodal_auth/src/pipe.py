from abc import ABC, abstractmethod
import copy
import warnings
import librosa
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Any, Dict, Literal, Optional, Callable, Tuple, Mapping
from torchvision import transforms
import os
import torchaudio
import onnxruntime
from facenet_pytorch import MTCNN
import numpy as np

from synthweave.fusion.base import BaseFusion
from synthweave.pipeline.base import BasePipeline
from .iil import Decomposer, CholeskyWhitening


# ====================================
#             VISUAL BRANCH
# ====================================
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlockIR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


class Bottleneck:
    def __init__(self, in_channel, depth, stride):
        self.in_channel = in_channel
        self.depth = depth
        self.stride = stride


def get_blocks(num_layers=50):
    return [
        get_block(64, 64, 3),
        get_block(64, 128, 4),
        get_block(128, 256, 14),
        get_block(256, 512, 3),
    ]


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()


class Backbone(nn.Module):
    def __init__(self, input_size=(112, 112), num_layers=50, mode="ir"):
        super(Backbone, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
        )

        blocks = get_blocks(num_layers)
        modules = [
            BasicBlockIR(b.in_channel, b.depth, b.stride)
            for block in blocks
            for b in block
        ]
        self.body = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )

        initialize_weights(self.modules())

    def forward(self, x):
        x = self.input_layer(x)
        for module in self.body:
            x = module(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)
        return output, norm


def IR_50(input_size=(112, 112)):
    return Backbone(input_size, 50, "ir")


def load_pretrained_model(path):
    # load model and pretrained statedict
    model = IR_50()
    statedict = torch.load(
        os.path.join(path, "adaface_ir50_ms1mv2.ckpt"), weights_only=False
    )["state_dict"]
    model_statedict = {
        key[6:]: val for key, val in statedict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_statedict)
    model.eval()
    return model


class AdaFace(nn.Module):
    def __init__(self, path: str, freeze=True):
        super(AdaFace, self).__init__()
        self._prepare_model(path, freeze)

    def _prepare_model(self, path, freeze):
        self.model = load_pretrained_model(path)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, images):
        """
        Expects images as torch.Tensor with shape [B, 3, H, W] and pixel values in [0, 1]
        """
        if not torch.is_tensor(images):
            raise ValueError("Input must be a PyTorch tensor.")

        # Normalize to match AdaFace training: mean=0.5, std=0.5
        images = (images - 0.5) / 0.5

        embeddings, _ = self.model(images)
        return embeddings

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)) * 100


class ReDimNet(nn.Module):
    def __init__(self, freeze=True):
        super(ReDimNet, self).__init__()
        self._prepare_model(freeze)

    def _prepare_model(self, freeze):
        self.model = torch.hub.load(
            repo_or_dir="IDRnD/ReDimNet",
            model="ReDimNet",
            model_name="b6",
            train_type="ptn",
            dataset="vox2",
            # force_reload=True
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, audios):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)

            # embeddings = []

            # for audio in audios:
            #     emb = self.model(audio)
            #     embeddings.append(emb.flatten())

            # embeddings = torch.stack(embeddings, dim=0)
            embeddings = self.model(audios)
            return embeddings

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)) * 100


class HeadPose:
    def __init__(self, dirpath):
        self.model_paths = [
            os.path.join(dirpath, "fsanet-1x1-iter-688590.onnx"),
            os.path.join(dirpath, "fsanet-var-iter-688590.onnx"),
        ]
        self.models = [
            onnxruntime.InferenceSession(model_path) for model_path in self.model_paths
        ]

    def __call__(self, image):
        image = image.permute(2, 0, 1).float()  # HWC to CHW
        image = self.transform(image)
        image = image / 255.0
        image = image.unsqueeze(0).numpy()
        yaw_pitch_roll_results = [
            model.run(["output"], {"input": image})[0] for model in self.models
        ]
        yaw, pitch, roll = np.mean(np.vstack(yaw_pitch_roll_results), axis=0)
        return yaw, pitch, roll

    def transform(self, image):
        trans = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.Normalize(mean=127.5, std=128.0)]
        )
        image = trans(image)
        return image


class ImagePreprocessor:
    def __init__(
        self,
        window_len: int = 4,
        step: int = 1,
        crop_face: bool = True,
        estimate_head_pose: bool = True,
        head_pose_dir: str = None,
        pad_mode: str = "repeat",  # 'repeat' or 'zeros'
        device: str = "cuda",
    ):
        self.window_len = window_len
        self.step = step
        self.crop_face = crop_face
        self.estimate_head_pose = estimate_head_pose
        self.pad_mode = pad_mode
        self.device = device

        if crop_face:
            self.face_detector = MTCNN(
                image_size=112,
                margin=0,
                post_process=False,
                keep_all=False,
                device=device,
            )

        if estimate_head_pose:
            self.head_pose_estimator = HeadPose(head_pose_dir)

        self.transform = transforms.Compose([transforms.Lambda(lambda x: x.float())])

    def __call__(self, video_input: torch.Tensor, fps: float) -> torch.Tensor:
        tensor = self._process_video(video_input, fps)
        return tensor

    def _crop_face(self, frame: np.ndarray) -> np.ndarray:
        # Returns a cropped face if detected, otherwise None.
        face_crop = self.face_detector(frame)
        return face_crop

    def _get_face_with_fallback(
        self, frame: np.ndarray, window: list, idx: int
    ) -> np.ndarray:
        face = self._crop_face(frame)
        if face is not None:
            return face, True

        # 1. try to find a frontal frame
        if self.estimate_head_pose:
            next_idx = idx + 1
            while next_idx < len(window):

                relative_frontal_idx = self._select_frontal_frame(
                    window[next_idx:]
                )  # return relative index
                if relative_frontal_idx is None:  # No more frames
                    break

                frontal_idx = next_idx + relative_frontal_idx

                fallback_frame = window[frontal_idx].numpy()
                face = self._crop_face(fallback_frame)

                if face is not None:
                    return face, True

                next_idx = frontal_idx + 1  # continue searching from last found index

        # 2. If estimate_head_pose is False, try a naive approach over all frames in window (except the original idx)
        else:
            for i, frame in enumerate(window):
                if i == idx:
                    continue  # already checked
                candidate = self._crop_face(frame.numpy())
                if candidate is not None:
                    return candidate, True

        # 3. If we exhaust all fallback options, return a failure
        return None, False

    def _check_frontal_face(self, image):
        yaw, pitch, roll = self.head_pose_estimator(image)
        return abs(yaw) < 30 and abs(pitch) < 30 and abs(roll) < 30

    def _select_frontal_frame(self, frames: list):
        # Returns the first frame with a frontal face
        for idx, frame in enumerate(frames):
            if self._check_frontal_face(frame):
                return idx
        else:
            return None  # No frontal face found

    # def _pad_window(self, window: torch.Tensor, target_length: int) -> torch.Tensor:
    #     T = window.shape[0]

    #     if T >= target_length:
    #         return window

    #     if self.pad_mode == 'repeat':
    #         repeat_factor = (target_length // T) + 1
    #         window = window.repeat(repeat_factor, 1, 1, 1)
    #         window = window[:target_length]

    #     elif self.pad_mode == 'zeros':
    #         padding = torch.zeros((target_length - T, *window.shape[1:]), dtype=window.dtype, device=window.device)
    #         window = torch.cat([window, padding], dim=0)

    # else:
    #     raise ValueError(f"Invalid padding mode: {self.pad_mode}")

    #     return window

    def _process_video(
        self, video_input: torch.Tensor, fps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # video_input: Tensor of shape (T, H, W, C) in uint8.
        fps = int(fps)
        num_frames = video_input.shape[0]

        if self.window_len == -1:  # use the entire video
            windows = [video_input]

        else:  # split video into windows
            frames_per_window = self.window_len * fps  # number of frames in a window
            step_frames = self.step * fps  # number of frames to skip between windows
            # usable_frames = (num_frames // fps) * fps # drop the last few frames if not enough for a window

            # process video in windows
            windows = [
                video_input[i : i + frames_per_window]
                for i in range(0, num_frames, step_frames)
            ]

        frames = []
        valid_mask = []
        for idx, window in enumerate(windows):
            is_valid = True
            # select frontal frame
            if self.estimate_head_pose:
                idx = self._select_frontal_frame(window)

                if idx is None:
                    frame = torch.zeros(
                        (3, 112, 112), dtype=torch.float32
                    )  # empty face
                    frames.append(frame)
                    valid_mask.append(False)
            else:
                idx = len(window) // 2

            frame = window[idx].numpy()

            # crop face
            if self.crop_face:
                cropped_face, found = self._get_face_with_fallback(frame, window, idx)

                if found:
                    frame = cropped_face
                else:
                    frame = torch.zeros(
                        (3, 112, 112), dtype=torch.float32
                    )  # empty face
                    is_valid = False

            # apply transform
            if is_valid:
                frame = self.transform(frame)
                frame = frame / 255.0

            frames.append(frame)

            # mask for valid frames
            if is_valid:
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        # tensor output and validity mask
        return torch.stack(frames, dim=0), torch.tensor(valid_mask, dtype=torch.bool)

class AudioMetric(ABC):
    @abstractmethod
    def __call__(self, audio_input: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def normalize_metric(self, metric: torch.Tensor) -> torch.Tensor:
        pass


class SNREstimator(AudioMetric):
    def __init__(
        self,
        frame_length=1024,  # Number of samples per frame for audio analysis
        hop_length=512,  # Number of samples between frames
        noise_percentile=10,  # Percentile threshold for noise estimation
        max_snr=30,  # Maximum SNR value when noise power is very low or zero
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.noise_percentile = noise_percentile
        self.max_snr = max_snr

    def normalize_metric(self, metric: torch.Tensor) -> torch.Tensor:
        """
        Normalize SNR values to be between 0 and 1
        """
        return torch.clamp(1 - (metric / self.max_snr), 0, 1)

    def __call__(self, audio_input: torch.Tensor) -> torch.Tensor:
        N, C, T = audio_input.shape
        results = []

        for n in range(N):
            for c in range(C):
                y = audio_input[n, c].numpy()
                frames = librosa.util.frame(
                    y, frame_length=self.frame_length, hop_length=self.hop_length
                )
                frame_energies = np.sum(frames**2, axis=0)
                noise_threshold = np.percentile(frame_energies, self.noise_percentile)
                noise_frames = frames[:, frame_energies <= noise_threshold]

                if noise_frames.size > 0:
                    noise_power = np.mean(noise_frames**2)
                    signal_power = np.mean(y**2)
                    snr = (
                        10 * np.log10(signal_power / noise_power)
                        if noise_power > 0
                        else self.max_snr
                    )
                else:
                    snr = self.max_snr

                results.append(snr)

        return torch.tensor(results).reshape(N, 1)
    
class AudioPreprocessor:
    def __init__(
        self,
        window_len: int = 4,
        step: int = 1,
        sample_rate: int = 16_000,
        max_len: int = 4,
        pad_mode: str = "repeat",  # 'repeat' or 'zeros'
        device: str = "cuda",
    ):
        self.window_len = window_len
        self.step = step
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.device = device
        self.pad_mode = pad_mode

    def __call__(self, audio_input: torch.Tensor, sr: int) -> torch.Tensor:
        tensor = self._process_audio(audio_input, sr)
        return tensor

    def _pad_audio(self, audio: torch.Tensor, len: int) -> torch.Tensor:
        data_len = audio.shape[0]

        if self.pad_mode == "repeat":
            repeats = len // data_len
            remainder = len % data_len
            audio = torch.cat([audio] * repeats + [audio[:remainder]])

        elif self.pad_mode == "zeros":
            pad_len = len - data_len
            audio = torch.cat([audio, torch.zeros(pad_len)])
        else:
            raise ValueError(f"Invalid padding mode: {self.pad_mode}")

        return audio

    def _process_audio(
        self, audio_input: torch.Tensor, sr: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio_input.numel() == 0:
            raise ValueError("No audio detected")

        # resample audio
        if sr != self.sample_rate:
            audio_input = torchaudio.functional.resample(
                waveform=audio_input, orig_freq=sr, new_freq=self.sample_rate
            )

        # convert to mono
        if audio_input.ndim == 2 and audio_input.shape[0] > 1:
            audio_input = audio_input.mean(dim=0)
        else:
            audio_input = audio_input.squeeze()

        total_samples = audio_input.shape[0]

        if self.window_len == -1:  # use the entire audio
            # pad or truncate to max_len
            max_samples = self.max_len * self.sample_rate
            if total_samples >= max_samples:
                audio_input = audio_input[:max_samples]
            else:
                audio_input = self._pad_audio(audio_input, max_samples)

            windows = [audio_input]

        else:  # split audio into windows
            window_samples = (
                self.window_len * self.sample_rate
            )  # number of samples in a window
            step_samples = (
                self.step * self.sample_rate
            )  # number of samples to skip between windows
            # usable_samples = (total_samples // self.sample_rate) * self.sample_rate # drop the last few samples if not enough for a window

            windows = []
            for i in range(0, total_samples, step_samples):
                window = audio_input[i : i + window_samples]
                if window.shape[0] < window_samples:  # pad window if too short
                    window = self._pad_audio(window, window_samples)
                windows.append(window)

        audios = torch.stack(windows, dim=0)

        if audios.dim() == 2:
            audios = audios.unsqueeze(1)  # add channel dimension

        valid_mask = torch.ones(
            audios.shape[0], dtype=torch.bool
        )  # valid mask for all windows
        return audios, valid_mask


class MultiModalAuthPipeline(BasePipeline):
    """
    Pipeline for multimodal authentication with deepfake detection module.
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        fusion: BaseFusion,
        detection_head: Optional[nn.Module] = None,
        processors: Optional[Mapping[str, Callable[..., torch.Tensor]]] = None,
        freeze_backbone: bool = True,
        iil_mode: Literal["none", "crossdf", "friday", "whitening"] = "whitening"
    ):
        super(MultiModalAuthPipeline, self).__init__(
            models, fusion, detection_head, processors, freeze_backbone
        )
        
        self.iil_mode = iil_mode
        self.backbones_frozen = freeze_backbone
        audio_dim = self.fusion.input_dims.get("audio", 192)
        video_dim = self.fusion.input_dims.get("video", 512)
        
        out_dim = self.fusion.output_dim
        
        if self.iil_mode == "crossdf":
            self.decomposer = Decomposer(dim=out_dim)
            
        elif self.iil_mode == "whitening":
            self.audio_whitening = CholeskyWhitening(
                audio_dim, 
                mode='ZCA',
                eps=1e-8,  # Slightly higher for more aggressive decorrelation
                track_running_stats=True,
                momentum=0.05  # Slower adaptation, more stable
            )
            self.video_whitening = CholeskyWhitening(
                video_dim,
                mode='ZCA', 
                eps=1e-8,
                track_running_stats=True,
                momentum=0.05
            )
        
        self.fusion_reg = nn.LayerNorm(out_dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.fusion.dropout_p)

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        output = {}
        
        # Preprocess inputs
        inputs = self.preprocess(inputs)

        # Extract embeddings from each modality
        feats = self.extract_features(inputs)
        
        output['video'] = feats['video']
        output['audio'] = feats['audio']
        
        if self.iil_mode == "whitening":
            # Feature whitening
            if self.backbones_frozen:
                with torch.no_grad():
                    feats['audio'] = self.audio_whitening(feats['audio'])
                    feats['video'] = self.video_whitening(feats['video'])
                    
                    output['audio_w'] = feats['audio']
                    output['video_w'] = feats['video']
        
        # Project and fuse embeddings into one vector
        fus_out: dict = self.fusion(
            {modality: feats[modality] for modality in self.fusion.modalities},
            output_projections=True,
        )
        
        output["audio_proj"] = fus_out["audio_proj"]
        output["video_proj"] = fus_out["video_proj"]
        
        z_f = self.dropout(self.relu(self.fusion_reg(fus_out["embedding"])))
        
        if self.iil_mode == "crossdf":
            z_f, z_os, _ = self.decomposer(z_f)
            output["id_embedding"] = z_os
            
        output["embedding"] = z_f
        
        # Pass the fused embeddings to the head
        head_out = self.downstream_pass(output)
        
        output["logits"] = head_out["logits"]
        return output

    def verify(self, inputs: Dict[str, Any]) -> torch.Tensor:
        similarities = {}

        for modality in self.fusion.modalities:
            embedding = inputs[modality]
            refference = inputs[modality + "_ref"]

            sim = self.feature_extractors[modality].compute_similarities(
                embedding, refference
            )
            similarities[modality] = sim

        return similarities
