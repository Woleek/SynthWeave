import sys
sys.path.append('../../..')

from abc import ABC, abstractmethod
import warnings
import librosa
import torch
import torch.nn as nn
from typing import Any, Dict, Literal, Optional, Callable, Tuple, Mapping
from torchvision import transforms
import os
import torchaudio
import onnxruntime
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image
from synthweave.fusion.base import BaseFusion
from synthweave.pipeline.base import BasePipeline
from collections import namedtuple
from .iil import Decomposer, CholeskyWhitening
import torch.nn.functional as F

# ====================================
#             VISUAL BRANCH
# ====================================
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BasicBlockIR(nn.Module):
    """ BasicBlock for IRNet
    """
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_blocks(num_layers=100):
    if num_layers == 50:
        return [
            get_block(64, 64, 3),
            get_block(64, 128, 4),
            get_block(128, 256, 14),
            get_block(256, 512, 3),
        ]
    elif num_layers == 100:
        return [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError(
            "num_layers should be 50 or 100, but got {}".format(num_layers)
        )


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
    def __init__(self, input_size, num_layers, mode='ir'):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      nn.BatchNorm2d(64), nn.PReLU(64))
        blocks = get_blocks(num_layers)
        unit_module = BasicBlockIR
        output_channel = 512

        self.output_layer = nn.Sequential(nn.BatchNorm2d(output_channel),
                                    nn.Dropout(0.4), Flatten(),
                                    nn.Linear(output_channel * 7 * 7, 512),
                                    nn.BatchNorm1d(512, affine=False))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        initialize_weights(self.modules())

    def forward(self, x):
        
        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        output = torch.div(x, norm)

        return output, norm


def IR_101(input_size=(112, 112)):
    return Backbone(input_size, 100, "ir")

def IR_50(input_size=(112, 112)):
    return Backbone(input_size, 50, "ir")

def load_pretrained_model(path, model_type: str = "ir101", device: str = "cuda"):
    # load model and pretrained statedict
    if model_type == "ir50":
        model = IR_50()
        statedict = torch.load(
            os.path.join(path, "adaface_ir50_ms1mv2.ckpt"), weights_only=False, 
            map_location=torch.device(device)
        )["state_dict"]
    elif model_type == "ir101":
        model = IR_101()
        statedict = torch.load(
            os.path.join(path, "adaface_ir101_ms1mv3.ckpt"), weights_only=False,
            map_location=torch.device(device)
        )["state_dict"]
    model_statedict = {
        key[6:]: val for key, val in statedict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_statedict)
    model.eval()
    return model


class AdaFace(nn.Module):
    def __init__(self, path: str, freeze=True, model_type: str = "ir101", device: str = "cuda"):
        super(AdaFace, self).__init__()
        self._prepare_model(path, freeze, model_type, device)

    def _prepare_model(self, path, freeze, model_type, device):
        self.model = load_pretrained_model(path, model_type, device)

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
        embeddings, _ = self.model(images)
        return embeddings

    def compute_similarities(self, e_i, e_j):
        return e_i @ e_j.T


class QualityAdaFace(nn.Module):
    def __init__(self, path: str, freeze=True, model_type: Literal["ir101", "ir50"] = "ir101"):
        super(QualityAdaFace, self).__init__()
        original_model = AdaFace(path, freeze, model_type)
        self.input_layer = original_model.model.input_layer
        self.body = original_model.model.body
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, 1).cuda()

        ckpt_path = os.path.join(path, "adaface_sdd_fiqa_mod.pth")
        checkpoint = torch.load(ckpt_path)
        self.load_state_dict(checkpoint)
        if freeze:
            self.eval()
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        x = self.fc(x)
        return x


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
        models_dir: str = None,
        estimate_quality: bool = True,
        quality_model_type: Literal["ir101", "ir50"] = "ir101",
        pad_mode: str = "repeat",  # 'repeat' or 'zeros'
        device: str = "cuda",
    ):
        self.window_len = window_len
        self.step = step
        self.crop_face = crop_face
        self.estimate_head_pose = estimate_head_pose
        self.estimate_quality = estimate_quality
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
            self.head_pose_estimator = HeadPose(models_dir)
        
        if self.estimate_quality:
            self.quality_estimator = QualityAdaFace(
                path=os.path.join(models_dir), freeze=True, model_type=quality_model_type
            ).to(device)

        # self.transform = transforms.Compose([transforms.Lambda(lambda x: x.float())])
        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, video_input: torch.Tensor, fps: float) -> torch.Tensor:
        tensor = self._process_video(video_input, fps)
        return tensor

    def _crop_faces(self, frames: np.ndarray) -> np.ndarray:
        cropped_faces = []
        for img in frames:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_cropped = self.face_detector(img)
            if img_cropped is None:
                continue
            cropped_faces.append(img_cropped)
        return torch.stack(cropped_faces) if cropped_faces else None


    def _check_frontal_face(self, image):
        yaw, pitch, roll = self.head_pose_estimator(image)
        return abs(yaw) < 20 and abs(pitch) < 20 and abs(roll) < 20

    def _select_frontal_frame(self, frames: list):
        # Returns the first frame with a frontal face
        frontal_frames = []
        for idx, frame in enumerate(frames):
            if self._check_frontal_face(frame):
                frontal_frames.append(frame.numpy())
        return frontal_frames if frontal_frames else None

    def _estimate_quality(self, frames: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if len(frames) > 10:
            step = len(frames) // 10
            frames = frames[::step]
        qualities = self.quality_estimator(frames.to(self.device))
        best_quality = torch.argmax(qualities).item()
        return frames[best_quality], qualities[best_quality]

    def _process_video(
        self, video_input: torch.Tensor, fps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # video_input: Tensor of shape (T, H, W, C) in uint8.
        fps = int(fps)
        num_frames = video_input.shape[0]
        # input_video = input_video[:, :, :, ::-1]

        if self.window_len == -1:  # use the entire video
            windows = [video_input]

        else:  # split video into windows
            frames_per_window = self.window_len * fps  # number of frames in a window
            step_frames = self.step * fps  # number of frames to skip between windows

            # process video in windows
            windows = [
                video_input[i : i + frames_per_window]
                for i in range(0, num_frames, step_frames)
            ]

        final_frames = []
        qualities = []
        valid_mask = []
        for idx, window in enumerate(windows):
            is_valid = True
            # select frontal frame
            if self.estimate_head_pose:
                frames = self._select_frontal_frame(window)

                if frames is None:
                    frame = torch.zeros(
                        (3, 112, 112), dtype=torch.float32
                    )  # empty face
                    final_frames.append(frame)
                    valid_mask.append(False)
                    qualities.append(float('-inf'))
                    continue
                frames = np.array(frames)
            else:
                frames = window.numpy()

            # crop face
            if self.crop_face:
                cropped_faces = self._crop_faces(frames)

                if cropped_faces is not None:
                    frames = cropped_faces
                else:
                    frame = torch.zeros(
                        (3, 112, 112), dtype=torch.float32
                    )  # empty face
                    final_frames.append(frame)
                    valid_mask.append(False)
                    qualities.append(float('-inf'))
                    continue

            if self.estimate_quality:
                if not isinstance(frames, torch.Tensor):
                    frames = torch.tensor(frames)
                best_frame, quality = self._estimate_quality(frames)
            else:
                best_frame = frames[0]
                quality = float('-inf')

            best_frame = best_frame / 255.0
            best_frame = self.transform(best_frame)
            final_frames.append(best_frame)
            qualities.append(quality)
            valid_mask.append(True)

        return torch.stack(final_frames, dim=0), torch.tensor(valid_mask, dtype=torch.bool)#, torch.tensor(qualities, dtype=torch.float32)

# ====================================
#             AUDIO BRANCH
# ====================================

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
        e_i = F.normalize(e_i, p=2, dim=1)
        e_j = F.normalize(e_j, p=2, dim=1)
        cos = torch.matmul(e_i, e_j.T)
        cos_sim_rescaled = (cos + 1) / 2  # Rescale to [0, 1]
        return cos_sim_rescaled

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
        use_vad: bool = False, # VAD flag option
    ):
        self.window_len = window_len
        self.step = step
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.device = device
        self.pad_mode = pad_mode
        self.use_vad = use_vad

        if self.use_vad:
            self._load_vad_model()
            
    def _load_vad_model(self):
        # Load the Silero VAD model
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.get_speech_timestamps = utils[0]
        self.vad_model.to(self.device)

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
    
    def _process_voice_only(
        self, audio_window: torch.Tensor, timestamps: list
    ) -> torch.Tensor:
        """
        Processes an audio window to keep only voice-active sections and repeat them to match original length.
        Assumes mono audio with shape (1, samples).
        Args:
            audio_window: Tensor of shape (1, samples) for mono audio
            timestamps: List of dictionaries with 'start' and 'end' keys indicating voice sections
        Returns:
            Processed audio window with only voice sections, repeated to match original length
        """
        # For mono audio with shape (1, samples)
        window_length = audio_window.shape[1]

        # Extract voice-active sections
        voice_sections = []
        for ts in timestamps:
            start, end = ts["start"], min(ts["end"], window_length)
            voice_sections.append(audio_window[0, start:end])

        # Create result tensor
        result = torch.zeros_like(audio_window)

        if voice_sections:  # If we found voice sections
            # Concatenate all voice sections
            voice_only = torch.cat(voice_sections)

            # Repeat voice sections to fill the original length
            if voice_only.numel() > 0:
                repeats_needed = window_length // voice_only.numel() + 1
                repeated_voice = voice_only.repeat(repeats_needed)
                result[0, :] = repeated_voice[:window_length]
                return result

        # Return original if no processing could be done
        return audio_window

    def _apply_vad(self, audio_windows: torch.Tensor) -> list:
        """
        Apply Voice Activity Detection to each audio window.
        Args:
            audio_windows: Tensor of shape [num_windows, channels, samples]
        Returns:
            List of speech timestamps for each window
        """
        vad_results = []

        # Process each window
        for window_idx in range(audio_windows.shape[0]):
            # Get the current window and ensure it's in the right format for VAD
            window = audio_windows[window_idx]

            # Ensure the audio is in the correct format for Silero VAD (mono)
            if window.dim() > 1 and window.shape[0] > 1:
                window = window.mean(dim=0)

            # Ensure the tensor is [1, samples] as expected by the VAD model
            if window.dim() == 1:
                window = window.unsqueeze(0)

            # Move to the same device as the VAD model
            window = window.to(self.device)

            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                window, self.vad_model, sampling_rate=self.sample_rate
            )

            vad_results.append(speech_timestamps)

        return vad_results

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
        
        # Apply VAD if enabled
        vad_info = None
        if self.use_vad:
            vad_info = self._apply_vad(audios)

            # Update valid_mask based on VAD results
            for i, timestamps in enumerate(vad_info):
                if not timestamps:  # No speech detected in this window
                    valid_mask[i] = False
                else:
                    # Process window to keep only voice sections and repeat them
                    audios[i] = self._process_voice_only(audios[i], timestamps)
        
        return audios, valid_mask


# ====================================
#             PIPELINE
# ====================================

class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.4, num_classes: int = 1):
        super().__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(input_dim, input_dim),
            # nn.LayerNorm(input_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(input_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )
        
        self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

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
        iil_mode: Literal["none", "crossdf", "friday", "whitening"] = "whitening",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(MultiModalAuthPipeline, self).__init__(
            models, fusion, detection_head, processors, freeze_backbone
        )
        
        self.device = device
        
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
        
        if self.processors is not None:
            # Drop invalid inputs
            valid_vids = inputs["video"][1]
            valid_auds = inputs["audio"][1]
            valid_pairs = valid_vids & valid_auds
            output["org_len"] = len(valid_vids)
            output["valid_len"] = valid_pairs.sum().item()
            
            # Filter inputs based on valid masks
            inputs["video"] = inputs["video"][0][valid_pairs]
            inputs["audio"] = inputs["audio"][0][valid_pairs]
        
        # Ensure correct device
        for modality in inputs:
            inputs[modality] = inputs[modality].to(self.device)

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
