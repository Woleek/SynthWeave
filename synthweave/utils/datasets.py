from collections import defaultdict
import json
import numbers
import os
import h5py
from pathlib import Path
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
# class LAV_DF_Dataset(Dataset):
#     def __init__(
#         self, root: str, metadata_file: str, split: Literal["train", "dev", "test"]
#     ):
#         super().__init__()

#         self.root = root
#         self.split = split
#         self._load_metadata(os.path.join(self.root, metadata_file), split)

#     def _load_metadata(self, metadata_file: str, split: str):
#         metadata = read_json(metadata_file)

#         metadata = [sample for sample in metadata if sample["split"] == split]
#         self.metadata = metadata

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx: int) -> dict:
#         meta: dict = self.metadata[idx]
#         video, audio, info = read_video(os.path.join(self.root, meta["file"]))
#         meta.update(info)

#         return {"video": video, "audio": audio, "metadata": meta}


# class LAV_DF_DataModule(LightningDataModule):
#     def __init__(
#         self, root: str, metadata_file: str, batch_size: int = 32, num_workers: int = 0
#     ):
#         super().__init__()

#         self.root = root
#         self.metadata_file = metadata_file
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#     def setup(self, stage: str = None):
#         if stage == "fit" or stage is None:
#             self.train_dataset = LAV_DF_Dataset(self.root, self.metadata_file, "train")
#             self.val_dataset = LAV_DF_Dataset(self.root, self.metadata_file, "dev")

#         if stage == "test" or stage is None:
#             self.test_dataset = LAV_DF_Dataset(self.root, self.metadata_file, "test")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
#         )


# =============================================================================
#     DeepSpeak_v1_1 (https://huggingface.co/datasets/faridlab/deepspeak_v1_1)
# =============================================================================
class DeepSpeak_v1_1_Dataset(Dataset):
    def __init__(
        self,
        split: Literal["train", "dev", "test"],
        video_processor: Optional[Callable] = None,
        audio_processor: Optional[Callable] = None,
        mode: Literal["minimal", "full"] = "minimal",
    ):
        super().__init__()

        self.mode = mode
        self.split = split
        self._prepare_dataset(split)

        self.video_processor = video_processor
        self.audio_processor = audio_processor

        if self.mode == "minimal":
            self._prepare_label_encoders()

    def _prepare_label_encoders(self):
        self.encoders = {"label": LabelEncoder(), "av": LabelEncoder()}

        self.encoders["label"].fit(["0", "1"])
        self.encoders["av"].fit(["00", "01", "10", "11"])

    def _prepare_dataset(self, split: str):
        if split == "test":
            dataset = load_dataset(
                "faridlab/deepspeak_v1_1", trust_remote_code=True, split=split
            )
            unique_labels = dataset.unique("type")
            class_label = ClassLabel(names=sorted(unique_labels))
            dataset = dataset.cast_column("type", class_label)
            self.dataset = dataset
        else:
            dataset = load_dataset(
                "faridlab/deepspeak_v1_1", trust_remote_code=True, split="train"
            )
            unique_labels = dataset.unique("type")
            class_label = ClassLabel(names=sorted(unique_labels))
            dataset = dataset.cast_column("type", class_label)
            split_dataset = dataset.train_test_split(
                test_size=0.1, seed=42, shuffle=True, stratify_by_column="type"
            )
            if split == "train":
                self.dataset = split_dataset["train"]
            elif split == "dev":
                self.dataset = split_dataset["test"]

    def __len__(self):
        return len(self.dataset)

    def _extract_minimal_metadata(self, meta: dict) -> dict:
        video_type = meta.get("type")

        if video_type == "fake":
            label = "1"

            # Determine AV label
            if meta.get("kind") == "face-swap":
                av = "01"
            elif meta.get("kind") == "lip-sync":
                if meta.get("recording-target-ai-generated") == True:
                    av = "11"
                else:
                    av = "10"

            # Extract identity info
            id_source = meta.get("identity-source")
            id_target = meta.get("identity-target")

        elif video_type == "real":
            label = "0"

            # Real audio & real video
            av = "00"

            # Extract identity
            id_source = meta.get("identity")
            id_target = id_source

        # encode labels
        label = self.encoders["label"].transform([label])[0]
        av = self.encoders["av"].transform([av])[0]

        return {
            "label": label,
            "av": av,
            "id_source": id_source,  # Identify of the person who created the video
            "id_target": id_target,  # Identity of the person who appears in the video
        }

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        video, audio, info = read_video(sample["video-file"])
        type = self.dataset.features["type"].int2str(sample["type"])
        meta: dict = (
            sample["metadata-fake"] if type == "fake" else sample["metadata-real"]
        )
        meta["type"] = type
        meta["file"] = sample["video-file"]
        meta.update(info)

        if self.video_processor:
            vid_fps = meta["video_fps"]
            video, valid_vid_seg = self.video_processor(video, vid_fps)

        if self.audio_processor:
            aud_sr = meta["audio_fps"]
            audio, valid_aud_seg = self.audio_processor(audio, aud_sr)

        if self.audio_processor and self.video_processor:
            # ensure same number of windows for video and audio
            min_len = min(video.shape[0], audio.shape[0])
            video = video[:min_len]
            valid_vid_seg = valid_vid_seg[:min_len]
            audio = audio[:min_len]
            valid_aud_seg = valid_aud_seg[:min_len]

            # keep only valid segments
            valid_joint = valid_vid_seg & valid_aud_seg
            valid_indices = valid_joint.nonzero(as_tuple=False).squeeze(-1)

            video = video[valid_indices]
            audio = audio[valid_indices]

        if self.mode == "minimal":
            meta = self._extract_minimal_metadata(meta)

        return {"video": video, "audio": audio, "metadata": meta}


class DeepSpeak_v1_1_DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        sample_mode: Literal["single", "sequence"] = "single",
        dataset_kwargs: dict = {},
        encode_ids: bool = True,
        # BATCH BALANCING
        clip_mode: Optional[Literal["id", "idx"]] = None,
        clip_to: Literal["min"] | int = "min",  # 'max' if padding will be added
        clip_selector: Literal["first", "random"] = "random",
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
            self.train_dataset = DeepSpeak_v1_1_Dataset("train", **self.dataset_kwargs)
            self.val_dataset = DeepSpeak_v1_1_Dataset("dev", **self.dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = DeepSpeak_v1_1_Dataset("test", **self.dataset_kwargs)

    def _collate_fn(self, batch: list) -> dict:
        # 1. Single mode - each window is a separate sample
        # NOTE: Need to ensure that some ID wont dominate batch - e.g. when some videos are much longer than others
        # TRY 1: set max number of samples per ID/video
        # TRY 2: clip to min number of samples per ID/video in current batch

        if self.sample_mode == "single":
            # Helper: build dict of lists based on grouping key
            def group_by_key(samples, key_fn):
                grouped = defaultdict(list)
                for i, sample in enumerate(samples):
                    vid_windows = sample["video"]
                    aud_windows = sample["audio"]
                    meta = sample["metadata"]
                    key = key_fn(meta, i)
                    grouped[key].append((vid_windows, aud_windows, meta))
                return grouped

            # Choose grouping strategy
            if self.clip_mode == "id":
                key_fn = lambda meta, i: meta["id_source"]
            elif self.clip_mode == "idx":
                key_fn = lambda meta, i: i
            else:
                key_fn = lambda meta, i: "default"

            grouped = group_by_key(batch, key_fn)

            video_stacks, audio_stacks, meta_flat_list = [], [], []

            for key, group in grouped.items():
                vids, auds, metas = [], [], []

                for vid_windows, aud_windows, meta in group:
                    vids.extend(vid_windows)
                    auds.extend(aud_windows)
                    metas.extend([meta] * len(vid_windows))

                # Clipping
                if self.clip_to == "min":
                    clip_val = min(len(vids), len(auds))
                elif isinstance(self.clip_to, int):
                    clip_val = min(self.clip_to, len(vids))
                else:
                    clip_val = len(vids)

                if self.clip_selector == "random" and clip_val < len(vids):
                    selected = np.random.choice(len(vids), clip_val, replace=False)
                    selected.sort()
                else:
                    selected = list(range(clip_val))

                if clip_val > 0:
                    video_stacks.append(torch.stack([vids[i] for i in selected]))
                    audio_stacks.append(torch.stack([auds[i] for i in selected]))
                    meta_flat_list.extend([metas[i] for i in selected])

            if not video_stacks or not audio_stacks:
                return None  # skip empty batch

            videos = torch.cat(video_stacks, dim=0)
            audios = torch.cat(audio_stacks, dim=0)

            # Organize metadata into dict of lists
            metas = {k: [meta[k] for meta in meta_flat_list] for k in meta_flat_list[0]}

        # 2. Sequence mode - each sequence of windows from the same video is a separate sample
        # NOTE: Need to handle various lenghts of sequences
        # TRY 1: clip/pad to min/max length with repeat/zeros (can cause issues with methods analyzing transitions)
        elif self.sample_mode == "sequence":
            raise NotImplementedError("Sequence mode not implemented yet")

        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")

        # Encode IDs if enabled
        if self.encode_ids:
            ids = metas["id_source"] + metas["id_target"]
            self.id_encoder.fit(ids)
            metas["id_source"] = self.id_encoder.transform(metas["id_source"])
            metas["id_target"] = self.id_encoder.transform(metas["id_target"])
            self.id_encoder = LabelEncoder()  # Reset for next batch

        # Tensorize metadata fields where appropriate
        for k, v_list in metas.items():
            if isinstance(v_list[0], (int, np.integer, numbers.Integral)):
                metas[k] = torch.as_tensor(v_list, dtype=torch.long)
            elif isinstance(v_list[0], (float, np.floating, numbers.Real)):
                metas[k] = torch.as_tensor(v_list, dtype=torch.float)
            else:
                metas[k] = v_list  # keep as list of strings or other types

        return {"video": videos, "audio": audios, "metadata": metas}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )


class DeepSpeak_v1_1_Dataset_prep(Dataset):
    def __init__(
        self,
        split: Literal["train", "dev", "test"],
        data_dir: str,
        sample_mode: Literal["single", "sequence"] = "single",
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_mode = sample_mode

        self.h5_path = self.data_dir / f"{split}.h5"
        self.index_path = self.data_dir / f"{split}_flat_index.json"

        self._prepare_label_encoders()
        self._load_index()

    def _prepare_label_encoders(self):
        self.encoders = {"label": LabelEncoder(), "av": LabelEncoder()}

        self.encoders["label"].fit(["0", "1"])
        self.encoders["av"].fit(["00", "01", "10", "11"])

    def _load_index(self):
        self.h5_file = h5py.File(self.h5_path, "r")

        if self.sample_mode == "single":
            with open(self.index_path, "r") as f:
                self.flat_index = json.load(f)
        elif self.sample_mode == "sequence":
            self.video_ids = list(self.h5_file.keys())

    def __len__(self):
        return (
            len(self.flat_index)
            if self.sample_mode == "single"
            else len(self.video_ids)
        )

    def __getitem__(self, idx: int):
        if self.sample_mode == "single":
            entry = self.flat_index[idx]
            vid = entry["sample_id"]
            window_idx = entry["window_idx"]

            video = torch.tensor(
                self.h5_file[vid]["video"][window_idx], dtype=torch.float32
            )
            audio = torch.tensor(
                self.h5_file[vid]["audio"][window_idx], dtype=torch.float32
            )
            metadata = json.loads(self.h5_file[vid].attrs["metadata"])

            # encode labels
            metadata["label"] = self.encoders["label"].transform([metadata["label"]])[0]
            metadata["av"] = self.encoders["av"].transform([metadata["av"]])[0]

        else:  # sequence
            vid = self.video_ids[idx]
            video = torch.tensor(self.h5_file[vid]["video"][:], dtype=torch.float32)
            audio = torch.tensor(self.h5_file[vid]["audio"][:], dtype=torch.float32)
            metadata = json.loads(self.h5_file[vid].attrs["metadata"])

            # encode labels
            metadata["label"] = self.encoders["label"].transform([metadata["label"]])[0]
            metadata["av"] = self.encoders["av"].transform([metadata["av"]])[0]

        return {"video": video, "audio": audio, "metadata": metadata}

    def close(self):
        self.h5_file.close()


class DeepSpeak_v1_1_DataModule_prep(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        sample_mode: Literal["single", "sequence"] = "single",
        encode_ids: bool = True,
        clip_mode: Optional[Literal["id", "idx"]] = None,
        clip_to: Literal["min"] | int = "min",
        clip_selector: Literal["first", "random"] = "random",
        dataset_kwargs: dict = {},
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_mode = sample_mode
        self.encode_ids = encode_ids
        self.clip_mode = clip_mode
        self.clip_to = clip_to
        self.clip_selector = clip_selector
        self.dataset_kwargs = dataset_kwargs

        if encode_ids:
            self.id_encoder = LabelEncoder()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = DeepSpeak_v1_1_Dataset_prep(
                "train", **self.dataset_kwargs
            )
            self.val_dataset = DeepSpeak_v1_1_Dataset_prep("dev", **self.dataset_kwargs)

        if stage == "test" or stage is None:
            self.test_dataset = DeepSpeak_v1_1_Dataset_prep(
                "test", **self.dataset_kwargs
            )

    def _collate_fn(self, batch):
        if self.sample_mode == "single":
            batch_vid = defaultdict(list)
            batch_aud = defaultdict(list)
            batch_meta = defaultdict(list)

            for i, sample in enumerate(batch):
                video, audio, meta = (
                    sample["video"],
                    sample["audio"],
                    sample["metadata"],
                )
                key = {"id": meta["id_source"], "idx": i}.get(self.clip_mode, "default")

                batch_vid[key].append(video)
                batch_aud[key].append(audio)
                batch_meta[key].extend([meta] * len([video]))

            if self.clip_to == "min":
                clip_val = min(len(v) for v in batch_vid.values())
            elif isinstance(self.clip_to, int):
                clip_val = self.clip_to
            else:
                clip_val = None  # no clipping

            video_stacks, audio_stacks, meta_flat = [], [], []

            for key in batch_vid:
                v = batch_vid[key]
                a = batch_aud[key]
                m = batch_meta[key]

                n = len(v)
                if clip_val is not None and clip_val < n:
                    if self.clip_selector == "random":
                        selected = np.random.choice(n, clip_val, replace=False)
                        selected.sort()
                    else:
                        selected = list(range(clip_val))

                    v = [v[i] for i in selected]
                    a = [a[i] for i in selected]
                    m = [m[i] for i in selected]

                if v and a:
                    video_stacks.append(torch.stack(v))
                    audio_stacks.append(torch.stack(a))
                    meta_flat.extend(m)

            if not video_stacks:
                return None  # Lightning will skip empty batches

            videos = torch.cat(video_stacks, dim=0)
            audios = torch.cat(audio_stacks, dim=0)

        elif self.sample_mode == "sequence":
            raise NotImplementedError("Sequence mode not implemented yet")

        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")

        # Organize metadata
        metas = {k: [m[k] for m in meta_flat] for k in meta_flat[0]}

        if self.encode_ids:
            all_ids = metas["id_source"] + metas["id_target"]
            self.id_encoder.fit(all_ids)
            metas["id_source"] = self.id_encoder.transform(metas["id_source"])
            metas["id_target"] = self.id_encoder.transform(metas["id_target"])
            self.id_encoder = LabelEncoder()  # Reset for next batch

        for k, v_list in metas.items():
            if isinstance(v_list[0], (int, np.integer, numbers.Integral)):
                metas[k] = torch.as_tensor(v_list, dtype=torch.long)
            elif isinstance(v_list[0], (float, np.floating, numbers.Real)):
                metas[k] = torch.as_tensor(v_list, dtype=torch.float)
            else:
                metas[k] = v_list

        return {"video": videos, "audio": audios, "metadata": metas}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def teardown(self, stage=None):
        for ds in ["train_dataset", "val_dataset", "test_dataset"]:
            if hasattr(self, ds):
                getattr(self, ds).close()


# =============================================================================
DATASET_MAP = {
    # "LAV-DF": LAV_DF_Dataset,
    "DeepSpeak_v1_1": {
        "original": DeepSpeak_v1_1_Dataset,
        "preprocessed": DeepSpeak_v1_1_Dataset_prep,
    },
}

DatasetType = Literal["LAV-DF", "DeepSpeak_v1_1"]


def get_dataset(dataset_type: DatasetType, **kwargs) -> Dataset:
    preprocessed = kwargs.get("preprocessed", False)
    if preprocessed:
        del kwargs["preprocessed"]

    if dataset_type not in DATASET_MAP:
        raise ValueError(
            f"Invalid dataset type: {dataset_type}, must be one of {list(DATASET_MAP.keys())}"
        )

    return DATASET_MAP[dataset_type]["preprocessed" if preprocessed else "original"](
        **kwargs
    )


DATAMODULE_MAP = {
    # "LAV-DF": LAV_DF_DataModule,
    "DeepSpeak_v1_1": {
        "original": DeepSpeak_v1_1_DataModule,
        "preprocessed": DeepSpeak_v1_1_DataModule_prep,
    },
}


def get_datamodule(dataset_type: DatasetType, **kwargs) -> LightningDataModule:
    preprocessed = kwargs.get("dataset_kwargs", {}).get("preprocessed", False)
    if preprocessed:
        del kwargs["dataset_kwargs"]["preprocessed"]

    if dataset_type not in DATAMODULE_MAP:
        raise ValueError(
            f"Invalid dataset type: {dataset_type}, must be one of {list(DATAMODULE_MAP.keys())}"
        )

    return DATAMODULE_MAP[dataset_type]["preprocessed" if preprocessed else "original"](
        **kwargs
    )
