import warnings
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Callable, Tuple, Mapping
from torchvision import transforms
import os
import torchaudio
import onnxruntime
from facenet_pytorch import MTCNN
import numpy as np
import cv2
from PIL import Image

from synthweave.fusion.base import BaseFusion
from synthweave.pipeline.base import BasePipeline

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

def IR_101(input_size=(112, 112)):
    return Backbone(input_size, 100, "ir")


def load_pretrained_model(path):
    # load model and pretrained statedict
    model = IR_101()
    statedict = torch.load(
        os.path.join(path, "adaface_ir101_ms1mv3.ckpt"), weights_only=False
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
        # images = (images - 0.5) / 0.5

        embeddings, _ = self.model(images)
        return embeddings

    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)) * 100


class QualityAdaFace(nn.Module):
    def __init__(self, path: str, freeze=True):
        super(QualityAdaFace, self).__init__()
        original_model = AdaFace(path, freeze)
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
        models_dir: str = None,
        estimate_quality: bool = True,
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
                path=os.path.join(models_dir), freeze=True
            )

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
        return abs(yaw) < 30 and abs(pitch) < 30 and abs(roll) < 30

    def _select_frontal_frame(self, frames: list):
        # Returns the first frame with a frontal face
        frontal_frames = []
        for idx, frame in enumerate(frames):
            if self._check_frontal_face(frame):
                frontal_frames.append(frame.numpy())
        return frontal_frames if frontal_frames else None

    def _estimate_quality(self, frames: torch.Tensor) -> Tuple[torch.Tensor, float]:
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

                if cropped_faces:
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

        return torch.stack(final_frames, dim=0), torch.tensor(valid_mask, dtype=torch.bool), torch.tensor(qualities, dtype=torch.float32)


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
    """Pipeline for multimodal authentication with deepfake detection module.

    Implements a complete pipeline workflow in the following sequence:
    1. Video and audio inputs (full or windowed)
    3. Preprocessing (face detection and alignment, voice activity detection and speech fragments extraction)
    4. Uni-modal feature extraction
    5. Fusion
    6. Decision (2x verification + deepfake detection)
    7. Hard voting output
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        fusion: BaseFusion,
        detection_head: Optional[nn.Module] = None,
        processors: Optional[Mapping[str, Callable[..., torch.Tensor]]] = None,
        freeze_backbone: bool = True,
    ):
        super(MultiModalAuthPipeline, self).__init__(
            models, fusion, detection_head, processors, freeze_backbone
        )

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        outs = super().forward(inputs, output_feats=True, output_projections=True)
        return outs

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
