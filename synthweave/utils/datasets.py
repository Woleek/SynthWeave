import os
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from datasets import load_dataset, ClassLabel

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
        split: Literal["train", "dev", "test"]
    ):
        super().__init__()
        
        self.split = split
        self._prepare_dataset(split)
            
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
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        video, audio, info = read_video(sample['video-file'])
        type = self.dataset.features['type'].int2str(sample['type'])
        meta: dict = sample['metadata-fake'] if type == 'fake' else sample['metadata-real']
        meta['type'] = type
        meta['file'] = sample['video-file']
        meta.update(info)
        
        return {
            "video": video,
            "audio": audio,
            "metadata": meta
        }
        
class DeepSpeak_v1_DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DeepSpeak_v1_Dataset("train")
            self.val_dataset = DeepSpeak_v1_Dataset("dev")
            
        if stage == "test" or stage is None:
            self.test_dataset = DeepSpeak_v1_Dataset("test")
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    
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
