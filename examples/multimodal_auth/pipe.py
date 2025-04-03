import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Callable, Tuple, Union, Mapping, List

from synthweave.fusion.base import BaseFusion
from synthweave.pipeline.base import BasePipeline

import torch
from torchvision import transforms
import torchaudio
from facenet_pytorch import MTCNN
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import numpy as np
import torch.nn as nn

class ImagePreprocessor:
    def __init__(
        self, 
        window_len: int = 4, 
        step: int = 1, 
        crop_face: bool = True,
        pad_mode: str = 'repeat', # 'repeat' or 'zeros'
        device: str = 'cuda'
    ):
        self.window_len = window_len
        self.step = step
        self.crop_face = crop_face
        self.pad_mode = pad_mode
        self.device = device
        
        if crop_face:
            self.face_detector = MTCNN(
                image_size=112, 
                margin=0, 
                post_process=False, 
                keep_all=False, 
                device=device
            )
            
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float())
        ])
        
    def __call__(self, video_input: torch.Tensor, fps: float) -> torch.Tensor:
        tensor = self._process_video(video_input, fps)
        return tensor
    
    def _crop_face(self, frame: np.ndarray) -> np.ndarray:
        # Returns a cropped face if detected, otherwise None.
        face_crop = self.face_detector(frame)
        return face_crop
    
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
        
    def _process_video(self, video_input: torch.Tensor, fps: float) -> torch.Tensor:
        # video_input: Tensor of shape (T, H, W, C) in uint8.
        fps = int(fps)
        num_frames = video_input.shape[0]
        
        if self.window_len == -1: # use the entire video
            windows = [video_input]
            
        else: # split video into windows
            frames_per_window = self.window_len * fps # number of frames in a window
            step_frames = self.step * fps # number of frames to skip between windows
            # usable_frames = (num_frames // fps) * fps # drop the last few frames if not enough for a window
            
            # process video in windows
            windows = [video_input[i:i+frames_per_window] for i in range(0, num_frames, step_frames)]
        
        frames = []
        for idx, window in enumerate(windows):
            # select middle frame 
            # TODO: Replace with most frontal face frame
            frame = window[len(window) // 2].numpy()
            
            # crop face
            if self.crop_face:
                face = self._crop_face(frame)
                if face is None:
                    print(f"No face detected in window {idx}. Skipping...")
                    continue
                else:
                    frame = face
              
            # apply transform  
            frame = self.transform(frame)
            frame = frame / 255.0
            frames.append(frame)
            
        return torch.stack(frames, dim=0)
    
    
class AudioPreprocessor():
    def __init__(
        self,
        window_len: int = 4,
        step: int = 1,
        sample_rate: int = 16_000,
        max_len: int = 4,
        pad_mode: str = 'repeat', # 'repeat' or 'zeros'
        device: str = 'cuda'
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
        
        if self.pad_mode == 'repeat':
            repeats = len // data_len
            remainder = len % data_len
            audio = torch.cat([audio] * repeats + [audio[:remainder]])
            
        elif self.pad_mode == 'zeros':
            pad_len = len - data_len
            audio = torch.cat([audio, torch.zeros(pad_len)])
        else:
            raise ValueError(f"Invalid padding mode: {self.pad_mode}")
        
        return audio
    
    def _process_audio(self, audio_input: torch.Tensor, sr: int) -> torch.Tensor:
        if audio_input.numel() == 0:
            print("No audio detected. Skipping...")
            return None
        
        # resample audio
        if sr != self.sample_rate:
            audio_input = torchaudio.functional.resample(
                waveform=audio_input,
                orig_freq=sr,
                new_freq=self.sample_rate
            )
            
        # convert to mono
        if audio_input.ndim == 2 and audio_input.shape[0] > 1:
            audio_input = audio_input.mean(dim=0)
        else:
            audio_input = audio_input.squeeze()
            
        total_samples = audio_input.shape[0]
        
        if self.window_len == -1: # use the entire audio
            # pad or truncate to max_len
            max_samples = self.max_len * self.sample_rate
            if total_samples >= max_samples:
                audio_input = audio_input[:max_samples]
            else:
                audio_input = self._pad_audio(audio_input, max_samples)
            
            windows = [audio_input]
            
        else: # split audio into windows
            window_samples = self.window_len * self.sample_rate # number of samples in a window
            step_samples = self.step * self.sample_rate # number of samples to skip between windows
            # usable_samples = (total_samples // self.sample_rate) * self.sample_rate # drop the last few samples if not enough for a window
            
            windows = []
            for i in range(0, total_samples, step_samples):
                window = audio_input[i:i+window_samples]
                if window.shape[0] < window_samples: # pad window if too short
                    window = self._pad_audio(window, window_samples)
                windows.append(window)
            
        audios = torch.stack(windows, dim=0)
        
        if audios.dim() == 2:
            audios = audios.unsqueeze(1) # add channel dimension
        
        return audios
            

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
        super(MultiModalAuthPipeline, self).__init__(models, fusion, detection_head, processors, freeze_backbone)
        
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        outs = super().forward(inputs, output_feats=True)
        return outs
    
    def verify(self, inputs: Dict[str, Any]) -> torch.Tensor:
        similarities = {}
        
        for modality in self.fusion.modalities:
            embedding = inputs[modality]
            refference = inputs[modality+'_ref']
            
            sim = self.feature_extractors[modality].compute_similarities(embedding, refference)
            similarities[modality] = sim
            
        return similarities
            
            