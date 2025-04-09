import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Callable, Tuple, Mapping
from torchvision import transforms
import os
import torchaudio
import onnxruntime
from facenet_pytorch import MTCNN
import numpy as np

from synthweave.fusion.base import BaseFusion
from synthweave.pipeline.base import BasePipeline

class HeadPose:
    def __init__(self, dirpath):
        self.model_paths = [os.path.join(dirpath, "fsanet-1x1-iter-688590.onnx"), os.path.join(dirpath, "fsanet-var-iter-688590.onnx")]
        self.models = [onnxruntime.InferenceSession(model_path) for model_path in self.model_paths]
    
    def __call__(self, image):
        image = image.permute(2, 0, 1).float() # HWC to CHW
        image = self.transform(image)
        image = image / 255.0
        image = image.unsqueeze(0).numpy()
        yaw_pitch_roll_results = [
            model.run(["output"], {"input": image})[0] for model in self.models
        ]
        yaw, pitch, roll = np.mean(np.vstack(yaw_pitch_roll_results), axis=0)
        return yaw, pitch, roll
    
    def transform(self, image):
        trans = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.Normalize(mean=127.5,std=128.0)
        ])
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
        pad_mode: str = 'repeat', # 'repeat' or 'zeros'
        device: str = 'cuda'
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
                device=device
            )
            
        if estimate_head_pose:
            self.head_pose_estimator = HeadPose(head_pose_dir)
            
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
    
    def _get_face_with_fallback(self, frame: np.ndarray, window: list, idx: int) -> np.ndarray:
        face = self._crop_face(frame)
        if face is not None:
            return face, True
        
        # 1. try to find a frontal frame
        if self.estimate_head_pose:
            next_idx = idx + 1
            while next_idx < len(window):
                
                relative_frontal_idx  = self._select_frontal_frame(window[next_idx:]) # return relative index
                if relative_frontal_idx is None: # No more frames
                    break
                
                frontal_idx = next_idx + relative_frontal_idx
                
                fallback_frame = window[frontal_idx].numpy()
                face = self._crop_face(fallback_frame)
                
                if face is not None:
                    return face, True
            
                next_idx = frontal_idx + 1 # continue searching from last found index

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
            return None # No frontal face found
    
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
        
    def _process_video(self, video_input: torch.Tensor, fps: float) -> Tuple[torch.Tensor, torch.Tensor]:
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
        valid_mask = []
        for idx, window in enumerate(windows):
            is_valid = True
            # select frontal frame
            if self.estimate_head_pose:
                idx = self._select_frontal_frame(window)
                
                if idx is None:
                    frame = torch.zeros((3, 112, 112), dtype=torch.float32) # empty face
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
                    frame = torch.zeros((3, 112, 112), dtype=torch.float32) # empty face
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
    
    def _process_audio(self, audio_input: torch.Tensor, sr: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio_input.numel() == 0:
            raise ValueError("No audio detected")
        
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
        
        valid_mask = torch.ones(audios.shape[0], dtype=torch.bool) # valid mask for all windows
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
            
            