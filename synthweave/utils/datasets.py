from collections import defaultdict
import numbers
import os
from typing import Callable, Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from datasets import load_dataset, ClassLabel
from sklearn.preprocessing import LabelEncoder

from synthweave.utils.tools import read_json, read_video

# =============================================================================
#          LAV-DF (https://huggingface.co/datasets/ControlNet/LAV-DF)
# =============================================================================
class LAV_DF_Dataset(Dataset):
    def __init__(
        self, 
        root: str, 
        metadata_file: str, 
        split: Literal["train", "dev", "test"]
    ):
        super().__init__()
        
        self.root = root
        self.split = split
        self._load_metadata(os.path.join(self.root, metadata_file), split)
            
    def _load_metadata(self, metadata_file: str, split: str):
        metadata = read_json(metadata_file)
        
        metadata = [sample for sample in metadata if sample["split"] == split]
        self.metadata = metadata
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        meta: dict = self.metadata[idx]
        video, audio, info = read_video(os.path.join(self.root, meta["file"]))
        meta.update(info)
        
        return {
            "video": video,
            "audio": audio,
            "metadata": meta
        }
        
class LAV_DF_DataModule(LightningDataModule):
    def __init__(
        self, 
        root: str, 
        metadata_file: str, 
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()
        
        self.root = root
        self.metadata_file = metadata_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = LAV_DF_Dataset(self.root, self.metadata_file, "train")
            self.val_dataset = LAV_DF_Dataset(self.root, self.metadata_file, "dev")
            
        if stage == "test" or stage is None:
            self.test_dataset = LAV_DF_Dataset(self.root, self.metadata_file, "test")
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
# =============================================================================
#     DeepSpeak_v1 (https://huggingface.co/datasets/faridlab/deepspeak_v1)
# =============================================================================
class DeepSpeak_v1_Dataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "dev", "test"],
        video_processor: Optional[Callable] = None,
        audio_processor: Optional[Callable] = None,
        mode: Literal['minimal', 'full'] = 'minimal'
    ):
        super().__init__()
        
        self.mode = mode
        self.split = split
        self._prepare_dataset(split)
        
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        
        if self.mode == 'minimal':
            self._prepare_label_encoders()
            
    def _prepare_label_encoders(self):
        self.encoders = {
            "label": LabelEncoder(),
            "av": LabelEncoder()
        }
        
        self.encoders["label"].fit(['0', '1'])
        self.encoders["av"].fit(['00', '01', '10', '11'])
            
    def _prepare_dataset(self, split: str):
        if split == "test":
            dataset = load_dataset("faridlab/deepspeak_v1", trust_remote_code=True, split=split)
            unique_labels = dataset.unique('type')
            class_label = ClassLabel(names=sorted(unique_labels))
            dataset = dataset.cast_column("type", class_label)
            self.dataset = dataset
        else:
            dataset = load_dataset("faridlab/deepspeak_v1", trust_remote_code=True, split="train")
            unique_labels = dataset.unique('type')
            class_label = ClassLabel(names=sorted(unique_labels))
            dataset = dataset.cast_column("type", class_label)
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True, stratify_by_column='type')
            if split == "train":
                self.dataset = split_dataset['train']
            elif split == "dev":
                self.dataset = split_dataset['test']
        
    def __len__(self):
        return len(self.dataset)
    
    def _extract_minimal_metadata(self, meta: dict) -> dict:
        video_type = meta.get("type")
        
        if video_type == "fake":
            label = '1'

            # Determine AV label
            if meta.get('kind') == "face-swap":
                av = '01'
            elif meta.get('kind') == "lip-sync":
                if meta.get('recording-target-ai-generated') == True:
                    av = '11'
                else:
                    av = '10'

            # Extract identity info
            id_source = meta.get("identity-source")
            id_target = meta.get("identity-target")

        elif video_type == "real":
            label = '0'

            # Real audio & real video
            av = '00'

            # Extract identity
            id_source = meta.get("identity")
            id_target = id_source
            
        # encode labels
        label = self.encoders["label"].transform([label])[0]
        av = self.encoders["av"].transform([av])[0]
            
        return {
            "label": label,
            "av": av,
            "id_source": id_source, # Identify of the person who created the video
            "id_target": id_target  # Identity of the person who appears in the video
        }   
        
    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        video, audio, info = read_video(sample['video-file'])
        type = self.dataset.features['type'].int2str(sample['type'])
        meta: dict = sample['metadata-fake'] if type == 'fake' else sample['metadata-real']
        meta['type'] = type
        meta['file'] = sample['video-file']
        meta.update(info)
        
        if self.video_processor:
            vid_fps = meta['video_fps']
            video = self.video_processor(video, vid_fps)
            
        if self.audio_processor:
            aud_sr = meta['audio_fps']
            audio = self.audio_processor(audio, aud_sr)
        
        if self.audio_processor and self.video_processor:
            # ensure same number of windows for video and audio
            min_len = min(video.shape[0], audio.shape[0])
            video = video[:min_len]
            audio = audio[:min_len]
            
        if self.mode == 'minimal':
            meta = self._extract_minimal_metadata(meta)
        
        return {
            "video": video,
            "audio": audio,
            "metadata": meta
        }
        
class DeepSpeak_v1_DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        sample_mode: Literal['single', 'sequence'] = 'single', 
        dataset_kwargs: dict = {},
        encode_ids: bool = True,
        
        # BATCH BALANCING
        clip_mode: Optional[Literal['id', 'idx']] = None,
        clip_to: Literal['min'] | int = 'min', # 'max' if padding will be added
        clip_selector: Literal['first', 'random'] = 'first',
        # pad_mode: Optional[Literal['repeat', 'zeros']] = None
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs
        self.sample_mode = sample_mode
        self.encode_ids = encode_ids
        if self.encode_ids:
            self.id_encoder = LabelEncoder()
        
        self.clip_to = clip_to
        self.clip_mode = clip_mode
        self.clip_selector = clip_selector
        
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DeepSpeak_v1_Dataset("train", **self.dataset_kwargs)
            self.val_dataset = DeepSpeak_v1_Dataset("dev", **self.dataset_kwargs)
            
        if stage == "test" or stage is None:
            self.test_dataset = DeepSpeak_v1_Dataset("test", **self.dataset_kwargs)
            
    def _collate_fn(self, batch: list) -> dict:
        # 1. Single mode - each window is a separate sample
        # NOTE: Need to ensure that some ID wont dominate batch - e.g. when some videos are much longer than others
        # TRY 1: set max number of samples per ID/video
        # TRY 2: clip to min number of samples per ID/video in current batch
        
        if self.sample_mode == 'single':
            
            if self.clip_mode == 'id': # Balance by ID
                batch_vid = defaultdict(list)
                batch_aud = defaultdict(list)
                batch_metas = defaultdict(list)

                for idx, sample in enumerate(batch):
                    vid_windows = sample["video"]
                    aud_windows = sample["audio"]
                    meta = sample["metadata"]

                    # store metadata, video and audio windows per ID (source) in batch
                    batch_vid[meta['id_source']].extend(vid_windows)
                    batch_aud[meta['id_source']].extend(aud_windows)
                    batch_metas[meta['id_source']].extend([meta for _ in range(len(vid_windows))])
                    
                if self.clip_to == 'min': # Clip to min number of samples per ID in batch
                    clip_val = min([len(windows) for windows in batch_vid.values()])
                elif isinstance(self.clip_to, int): # Clip to specific number of samples per ID in batch
                    clip_val = self.clip_to
                else:
                    raise ValueError(f"Clipping selected but the mode is invalid: {self.clip_to}")

                for id, windows in batch_vid.items():
                    batch_vid[id] = windows[:clip_val]
                    batch_aud[id] = batch_aud[id][:clip_val]
                    batch_metas[id] = batch_metas[id][:clip_val]
                    
            elif self.clip_mode == 'idx': # Balance by sample index
                batch_vid = defaultdict(list)
                batch_aud = defaultdict(list)
                batch_metas = defaultdict(list)
                
                for idx, sample in enumerate(batch):
                    vid_windows = sample["video"]
                    aud_windows = sample["audio"]
                    meta = sample["metadata"]
                    
                    # store metadata, video and audio windows per sample index in batch
                    batch_vid[idx].extend(vid_windows)
                    batch_aud[idx].extend(aud_windows)
                    batch_metas[idx].extend([meta for _ in range(len(vid_windows))])
                    
                if self.clip_to == 'min': # Clip to min number of samples per sample index in batch
                    clip_val = min([len(windows) for windows in batch_vid.values()])
                elif isinstance(self.clip_to, int): # Clip to specific number of samples per sample index in batch
                    clip_val = self.clip_to
                else:
                    raise ValueError(f"Clipping selected but the mode is invalid: {self.clip_to}")
                
                for idx, windows in batch_vid.items():
                    if self.clip_selector == 'first':
                        batch_vid[idx] = windows[:clip_val]
                        batch_aud[idx] = batch_aud[idx][:clip_val]
                        batch_metas[idx] = batch_metas[idx][:clip_val]
                    elif self.clip_selector == 'random':
                        selected_idx = np.random.choice(len(windows), clip_val, replace=False)
                        selected_idx.sort()
                        batch_vid[idx] = [windows[i] for i in selected_idx]
                        batch_aud[idx] = [batch_aud[idx][i] for i in selected_idx]
                        batch_metas[idx] = [batch_metas[idx][i] for i in selected_idx]
                    
            else: # No balancing
                batch_vid = defaultdict(list)
                batch_aud = defaultdict(list)
                batch_metas = defaultdict(list)
                
                for idx, sample in enumerate(batch):
                    vid_windows = sample["video"]
                    aud_windows = sample["audio"]
                    meta = sample["metadata"]
                    
                    batch_vid['default'].extend(vid_windows)
                    batch_aud['default'].extend(aud_windows)
                    batch_metas['default'].extend([meta for _ in range(len(vid_windows))])
                    
            videos = torch.cat([torch.stack(windows, dim=0) for windows in batch_vid.values()], dim=0)
            audios = torch.cat([torch.stack(windows, dim=0) for windows in batch_aud.values()], dim=0)
            flat_metas = [meta for meta_list in batch_metas.values() for meta in meta_list]
            metas = {k: [meta[k] for meta in flat_metas] for k in flat_metas[0]}
            
        # 2. Sequence mode - each sequence of windows from the same video is a separate sample
        # NOTE: Need to handle various lenghts of sequences 
        # TRY 1: clip/pad to min/max length with repeat/zeros (can cause issues with methods analyzing transitions)
        elif self.sample_mode == 'sequence':
            raise NotImplementedError("Sequence mode not implemented yet")
        
        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")
        
        if self.encode_ids:
            ids = metas['id_source'] + metas['id_target']
            self.id_encoder.fit(ids)
            
            metas['id_source'] = self.id_encoder.transform(metas['id_source'])
            metas['id_target'] = self.id_encoder.transform(metas['id_target'])
            
            # reset encoder to avoid encoding errors
            self.id_encoder = LabelEncoder()
            
        # tensorize metadata where possible
        for k, v_list in metas.items():
            if isinstance(v_list[0], (int, np.integer, numbers.Integral)):
                metas[k] = torch.as_tensor(v_list, dtype=torch.long)
            elif isinstance(v_list[0], (float, np.floating, numbers.Real)):
                metas[k] = torch.as_tensor(v_list, dtype=torch.float)
            else:
                metas[k] = v_list
        
        return {
            "video": videos,
            "audio": audios,
            "metadata": metas
        }
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self._collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self._collate_fn)
    
    
# =============================================================================
DATASET_MAP = {
    "LAV-DF": LAV_DF_Dataset,
    "DeepSpeak_v1": DeepSpeak_v1_Dataset
}

DatasetType = Literal["LAV-DF", "DeepSpeak_v1"]

def get_dataset(dataset_type: DatasetType, **kwargs) -> Dataset:
    if dataset_type not in DATASET_MAP:
        raise ValueError(f"Invalid dataset type: {dataset_type}, must be one of {list(DATASET_MAP.keys())}")
    
    return DATASET_MAP[dataset_type](**kwargs)

DATAMODULE_MAP = {
    "LAV-DF": LAV_DF_DataModule,
    "DeepSpeak_v1": DeepSpeak_v1_DataModule
}

def get_datamodule(dataset_type: DatasetType, **kwargs) -> LightningDataModule:
    if dataset_type not in DATAMODULE_MAP:
        raise ValueError(f"Invalid dataset type: {dataset_type}, must be one of {list(DATAMODULE_MAP.keys())}")
    
    return DATAMODULE_MAP[dataset_type](**kwargs)
