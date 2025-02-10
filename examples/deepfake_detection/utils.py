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
        device: str = 'cuda'
    ):
        self.window_len = window_len
        self.step = step
        self.crop_face = crop_face
        self.device = device
        
        if crop_face:
            self.face_detector = MTCNN(image_size=112, margin=0, post_process=False, keep_all=False, device=device)
            
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float())
        ])
        
    def __call__(self, video_input: torch.Tensor, fps: float) -> torch.Tensor:
        tensor = self._process_video(video_input, fps)
        return tensor
    
    def _crop_face(self, frame: np.ndarray) -> np.ndarray:
        face_crop = self.face_detector(frame)
        return face_crop
        
    def _process_video(self, video_input: torch.Tensor, fps: float) -> torch.Tensor:
        # video_input: Tensor of shape (T, H, W, C) in uint8.
        fps = int(fps)
        num_frames = video_input.shape[0]
        
        if self.window_len == -1: # use the entire video
            windows = [video_input]
            
        else: # split video into windows
            frames_per_window = self.window_len * fps # number of frames in a window
            step_frames = self.step * fps # number of frames to skip between windows
            usable_frames = (num_frames // fps) * fps # drop the last few frames if not enough for a window
            
            # process video in windows
            windows = [video_input[i:i+frames_per_window] for i in range(0, usable_frames, step_frames)]
        
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
    
    
class ArcFace(nn.Module):
    def __init__(self):
        super(ArcFace, self).__init__()
        self._prepare_model()
    
    def _prepare_model(self):
        self.model = get_model('buffalo_l', allow_download=True, download_zip=True)

        if self.model is None:
            app = FaceAnalysis(name="buffalo_l")
            app.prepare(ctx_id=0)
            self.model = get_model('buffalo_l', allow_download=True, download_zip=True)

        self.model.prepare(ctx_id=0)
    
    def forward(self, images):
        images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1) # (N, H, W, C)
        images_np = images_np.astype(np.float32)
        
        embeddings = []
        for img in images_np:
            emb = self.model.get_feat(img)
            embeddings.append(emb.flatten())
        
        embeddings = np.stack(embeddings, axis=0)
        return torch.tensor(embeddings)
    
    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)) * 100
    
    
class AudioPreprocessor():
    def __init__(
        self,
        window_len: int = 4,
        step: int = 1,
        sample_rate: int = 16_000,
        pad_mode: str = 'repeat', # 'repeat' or 'zeros'
        device: str = 'cuda'
    ):
        self.window_len = window_len
        self.step = step
        self.sample_rate = sample_rate
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
            windows = [audio_input]
            
        else: # split audio into windows
            window_samples = self.window_len * self.sample_rate # number of samples in a window
            step_samples = self.step * self.sample_rate # number of samples to skip between windows
            usable_samples = (total_samples // self.sample_rate) * self.sample_rate # drop the last few samples if not enough for a window
            
            windows = [audio_input[i:i+window_samples] for i in range(0, usable_samples, step_samples)]
            
            for idx, window in enumerate(windows):
                if window.shape[0] < window_samples:
                    window = self._pad_audio(window, window_samples)
                    windows[idx] = window
            
        audios = torch.stack(windows, dim=0)
        
        if audios.dim() == 2:
            audios = audios.unsqueeze(1) # add channel dimension
        
        return audios
    
class ReDimNet(nn.Module):
    def __init__(self):
        super(ReDimNet, self).__init__()
        self._prepare_model()
    
    def _prepare_model(self):
        self.model = torch.hub.load(
            repo_or_dir='IDRnD/ReDimNet', 
            model='ReDimNet',
            model_name='b6',
            train_type='ptn',
            dataset='vox2',
            force_reload=True
        )

    def forward(self, audios):
        embeddings = []
        
        # for audio in audios:
        #     emb = self.model(audio)
        #     embeddings.append(emb.flatten())
        
        # embeddings = torch.stack(embeddings, dim=0)
        embeddings = self.model(audios)
        return embeddings
    
    def compute_similarities(self, e_i, e_j):
        return np.dot(e_i, e_j.T) / (np.linalg.norm(e_i) * np.linalg.norm(e_j)) * 100