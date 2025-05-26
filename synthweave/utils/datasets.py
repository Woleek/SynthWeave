from collections import Counter, defaultdict
import json
import numbers
import os
import re
import h5py
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
from datasets import load_dataset, ClassLabel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from numpy.random import default_rng

from synthweave.utils.tools import read_audio, read_json, read_video


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
            ds = load_dataset(
                "faridlab/deepspeak_v1_1",
                trust_remote_code=True,
                split=split
            )
            unique_labels = ds.unique("type")
            class_label = ClassLabel(names=sorted(unique_labels))
            self.dataset = ds.cast_column("type", class_label)
        else: # train/dev
            ds = load_dataset(
                "faridlab/deepspeak_v1_1",
                trust_remote_code=True,
                split="train",
            )

            class_label = ClassLabel(names=sorted(ds.unique("type")))
            ds = ds.cast_column("type", class_label)
            
            def add_id(example):
                """Extract the *target* identity for every clip."""
                t = class_label.int2str(example["type"])
                if t == "fake":
                    example["id_target"] = example["metadata-fake"]["identity-target"]
                else:                       # real clip
                    example["id_target"] = example["metadata-real"]["identity"]
                return example
            
            ds = ds.map(add_id, desc="Add id_target column")
            groups = ds["id_target"]
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=0.1,       # 10 % of identities → dev
                random_state=42
            )
            train_idx, dev_idx = next(gss.split(X=groups, groups=groups))
            
            if split == "train":
                self.dataset = ds.select(train_idx)
            elif split == "dev":
                self.dataset = ds.select(dev_idx)

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


# class DeepSpeak_v1_1_DataModule(LightningDataModule):
#     def __init__(
#         self,
#         batch_size: int = 32,
#         num_workers: int = 0,
#         sample_mode: Literal["single", "sequence"] = "single",
#         dataset_kwargs: dict = {},
#         encode_ids: bool = True,
#         # BATCH BALANCING
#         clip_mode: Optional[Literal["id", "idx"]] = None,
#         clip_to: None | Literal["min"] | int = None, # None == 'max'
#         clip_selector: Literal["first", "random"] = "random",
#         # PADDING (sequence mode)
#         pad_mode: Optional[Literal['repeat', 'zeros']] = 'repeat',
#         seq_len: None | Literal["min"] | int = None, # None == 'max'
#     ):
#         super().__init__()

#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.dataset_kwargs = dataset_kwargs
#         self.sample_mode = sample_mode
#         self.encode_ids = encode_ids
#         if self.encode_ids:
#             self.id_encoder = LabelEncoder()

#         self.clip_to = clip_to
#         self.clip_mode = clip_mode
#         self.clip_selector = clip_selector
#         self.pad_mode = pad_mode
#         self.seq_len = seq_len

#     def setup(self, stage: str = None):
#         if stage == "fit" or stage is None:
#             self.train_dataset = DeepSpeak_v1_1_Dataset("train", **self.dataset_kwargs)
#             self.val_dataset = DeepSpeak_v1_1_Dataset("dev", **self.dataset_kwargs)

#         if stage == "test" or stage is None:
#             self.test_dataset = DeepSpeak_v1_1_Dataset("test", **self.dataset_kwargs)

#     def _collate_fn(self, batch: list) -> dict:
#         """
#         Balancing logic
#         ───────────────
#         1.  Groups are formed with 'clip_mode' key:
#             - clip_mode="id"  →  identity  (meta["id_source"])
#             - clip_mode="idx" →  each item by its original sample idx
#             - otherwise       →  single bucket ("default")
#         2.  For every group we keep at most *K* sequences, where *K* is defined by 'clip_to':
#             - clip_to="min"  →  K = min(group sizes in the batch)
#             - clip_to=<int>  →  K = clip_to
#             - clip_to=None   →  keep everything
#         3.  If a group has more than *K* clips, the clips are selected
#             according to `clip_selector`:
#             - clip_selector="first"  →  first *K* clips
#             - clip_selector="random" →  random *K* clips
#         4.  (Sequence mode) The remaining clips are then trimmed or padded in time to length L, so they can be stacked into one tensor. Controled by 'pad_mode':
#             - pad_mode="repeat"  →  repeat sequence to fill the time dimension
#             - pad_mode="zeros"   →  pad with zeros to fill the time dimension
#         and 'seq_len':
#             - seq_len="min"  →  L = min(len of sequences in the batch)
#             - seq_len=<int>  →  L = seq_len
#             - seq_len=None   →  keep full length of all sequences
        
#         """
#         def bucket_key(meta, idx):
#             if self.clip_mode == "id":
#                 return meta["id_source"]
#             if self.clip_mode == "idx":
#                 return idx
#             return "default"

#         def limit_bucket(items):
#             """Select at most *K* elements from a list according to clip_selector."""
#             if K is None or len(items) <= K:
#                 return items
#             if self.clip_selector == "random":
#                 sel = np.random.choice(len(items), K, replace=False)
#                 sel.sort()
#                 return [items[i] for i in sel]
#             return items[:K]  # "first"

#         def pad_or_trim(v, a, target_len):
#             """Return (video, audio) clipped or padded to *target_len* frames."""
#             cur_len = v.shape[0]
#             # trim ---------------------------------------------------------
#             if cur_len > target_len:
#                 if self.clip_selector == "random":
#                     start = np.random.randint(0, cur_len - target_len + 1)
#                     v, a = v[start : start + target_len], a[start : start + target_len]
#                 else:
#                     v, a = v[:target_len], a[:target_len]
#             # pad ----------------------------------------------------------
#             elif cur_len < target_len:
#                 pad_len = target_len - cur_len
#                 if self.pad_mode == "repeat":
#                     v_pad = v[-1:].repeat(pad_len, *([1] * (v.ndim - 1)))
#                     a_pad = a[-1:].repeat(pad_len, *([1] * (a.ndim - 1)))
#                 else:
#                     v_pad = torch.zeros((pad_len, *v.shape[1:]), dtype=v.dtype)
#                     a_pad = torch.zeros((pad_len, *a.shape[1:]), dtype=a.dtype)
#                 v, a = torch.cat([v, v_pad]), torch.cat([a, a_pad])
#             return v, a
        
#         # Construct buckets based on clip_mode
#         buckets = defaultdict(list)
#         for idx, sample in enumerate(batch):
#             buckets[bucket_key(sample["metadata"], idx)].append(sample)
            
#         # Decide how many clips to keep per bucket
#         if self.clip_to == "min":
#             K = min(len(v) for v in buckets.values())
#         elif isinstance(self.clip_to, int):
#             K = self.clip_to
#         else:
#             K = None  # keep all
            
#         if self.sample_mode == "single":
#             video_ts, audio_ts, meta_list = [], [], []
#             for bucket_items in buckets.values():
#                 # flatten windows inside every sample first
#                 windows = [
#                     (vw, aw, meta)
#                     for s in limit_bucket(bucket_items)
#                     for vw, aw, meta in zip(s["video"], s["audio"], [s["metadata"]] * len(s["video"]))
#                 ]

#                 # Select windows from buckets
#                 N = len(windows)
#                 clip_val = N if K is None else min(K, N) if isinstance(K, int) else K
#                 if self.clip_selector == "random" and clip_val < N:
#                     sel = np.random.choice(N, clip_val, replace=False)
#                     sel.sort()
#                     windows = [windows[i] for i in sel]
#                 else:
#                     windows = windows[:clip_val]

#                 # Stack windows
#                 if windows:
#                     v_stack = torch.stack([w[0] for w in windows])
#                     a_stack = torch.stack([w[1] for w in windows])
#                     video_ts.append(v_stack)
#                     audio_ts.append(a_stack)
#                     meta_list.extend([w[2] for w in windows])

#             if not video_ts:  # empty batch? → skip
#                 return None

#             videos = torch.cat(video_ts, 0)
#             audios = torch.cat(audio_ts, 0)

#         elif self.sample_mode == "sequence":
            
#             # Select clips from buckets
#             selected = [s for bucket in buckets.values() for s in limit_bucket(bucket)]
#             if not selected:
#                 return None
            
#             # Collect video/audio/metadata
#             vids, auds, meta_list, lengths = [], [], [], []
#             for s in selected:
#                 vids.append(s["video"])
#                 auds.append(s["audio"])
#                 meta_list.append(s["metadata"])
#                 lengths.append(s["video"].shape[0]) # number of windows
                
#             # Determine target temporal length
#             if self.seq_len == "min":
#                 L = min(lengths)
#             elif isinstance(self.seq_len, int):
#                 L = self.seq_len
#             else:  # seq_len is None → pad to longest
#                 L = max(lengths)

#             # Pad or trim sequences
#             vids_padded, auds_padded = zip(*(pad_or_trim(v, a, L) for v, a in zip(vids, auds)))
#             videos = torch.stack(list(vids_padded), 0)
#             audios = torch.stack(list(auds_padded), 0)
            
#         else:
#             raise ValueError(f"Invalid sample mode: {self.sample_mode}")
        
#         # Organize metadata into dict of lists
#         metas = {k: [m[k] for m in meta_list] for k in meta_list[0]}
        
#         if self.sample_mode == "sequence":
#             metas["length"] = torch.as_tensor(lengths, dtype=torch.long)

#         # Encode IDs if enabled
#         if self.encode_ids:
#             ids = metas["id_source"] + metas["id_target"]
#             self.id_encoder.fit(ids)
#             metas["id_source"] = self.id_encoder.transform(metas["id_source"])
#             metas["id_target"] = self.id_encoder.transform(metas["id_target"])
#             self.id_encoder = LabelEncoder()  # Reset for next batch

#         # Tensorize metadata fields where appropriate
#         for k, v_list in metas.items():
#             if isinstance(v_list[0], (int, np.integer, numbers.Integral)):
#                 metas[k] = torch.as_tensor(v_list, dtype=torch.long)
#             elif isinstance(v_list[0], (float, np.floating, numbers.Real)):
#                 metas[k] = torch.as_tensor(v_list, dtype=torch.float)
#             else:
#                 metas[k] = v_list  # keep as list of strings or other types

#         return {"video": videos, "audio": audios, "metadata": metas}

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#             collate_fn=self._collate_fn,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             collate_fn=self._collate_fn,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             collate_fn=self._collate_fn,
#         )


# class DeepSpeak_v1_1_Dataset_prep(Dataset):
#     def __init__(
#         self,
#         split: Literal["train", "dev", "test"],
#         data_dir: str,
#         sample_mode: Literal["single", "sequence"] = "single",
#     ):
#         self.data_dir = Path(data_dir)
#         self.split = split
#         self.sample_mode = sample_mode

#         self.h5_path = self.data_dir / f"{split}.h5"
#         self.index_path = self.data_dir / f"{split}_flat_index.json"

#         self._prepare_label_encoders()
#         self._load_index()

#     def _prepare_label_encoders(self):
#         self.encoders = {"label": LabelEncoder(), "av": LabelEncoder()}

#         self.encoders["label"].fit(["0", "1"])
#         self.encoders["av"].fit(["00", "01", "10", "11"])

#     def _load_index(self):
#         self.h5_file = h5py.File(self.h5_path, "r")

#         if self.sample_mode == "single":
#             with open(self.index_path, "r") as f:
#                 self.flat_index = json.load(f)
#         elif self.sample_mode == "sequence":
#             self.video_ids = list(self.h5_file.keys())

#     def __len__(self):
#         return (
#             len(self.flat_index)
#             if self.sample_mode == "single"
#             else len(self.video_ids)
#         )

#     def __getitem__(self, idx: int):
#         if self.sample_mode == "single":
#             entry = self.flat_index[idx]
#             vid = entry["sample_id"]
#             window_idx = entry["window_idx"]

#             video = torch.tensor(
#                 self.h5_file[vid]["video"][window_idx], dtype=torch.float32
#             )
#             audio = torch.tensor(
#                 self.h5_file[vid]["audio"][window_idx], dtype=torch.float32
#             )
#             metadata = json.loads(self.h5_file[vid].attrs["metadata"])

#             # encode labels
#             metadata["label"] = self.encoders["label"].transform([metadata["label"]])[0]
#             metadata["av"] = self.encoders["av"].transform([metadata["av"]])[0]

#         else:  # sequence
#             vid = self.video_ids[idx]
#             video = torch.tensor(self.h5_file[vid]["video"][:], dtype=torch.float32)
#             audio = torch.tensor(self.h5_file[vid]["audio"][:], dtype=torch.float32)
#             metadata = json.loads(self.h5_file[vid].attrs["metadata"])

#             # encode labels
#             metadata["label"] = self.encoders["label"].transform([metadata["label"]])[0]
#             metadata["av"] = self.encoders["av"].transform([metadata["av"]])[0]

#         return {"video": video, "audio": audio, "metadata": metadata}

#     def close(self):
#         self.h5_file.close()


# class DeepSpeak_v1_1_DataModule_prep(LightningDataModule):
#     def __init__(
#         self,
#         batch_size: int = 32,
#         num_workers: int = 0,
#         sample_mode: Literal["single", "sequence"] = "single",
#         encode_ids: bool = True,
#         clip_mode: Optional[Literal["id", "idx"]] = None,
#         clip_to: Literal["min"] | int = "min",
#         clip_selector: Literal["first", "random"] = "random",
#         pad_mode: Optional[Literal['repeat', 'zeros']] = 'repeat',
#         seq_len: None | Literal["min"] | int = None,
#         dataset_kwargs: dict = {},
#     ):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.sample_mode = sample_mode
#         self.encode_ids = encode_ids
#         self.clip_mode = clip_mode
#         self.clip_to = clip_to
#         self.clip_selector = clip_selector
#         self.pad_mode = pad_mode
#         self.seq_len = seq_len
#         self.dataset_kwargs = dataset_kwargs

#         if encode_ids:
#             self.id_encoder = LabelEncoder()

#     def setup(self, stage=None):
#         if stage == "fit" or stage is None:
#             self.train_dataset = DeepSpeak_v1_1_Dataset_prep(
#                 "train", **self.dataset_kwargs
#             )
#             self.val_dataset = DeepSpeak_v1_1_Dataset_prep("dev", **self.dataset_kwargs)

#         if stage == "test" or stage is None:
#             self.test_dataset = DeepSpeak_v1_1_Dataset_prep(
#                 "test", **self.dataset_kwargs
#             )

#     def _collate_fn(self, batch):
#         def bucket_key(meta, idx):
#             if self.clip_mode == "id":
#                 return meta["id_source"]
#             if self.clip_mode == "idx":
#                 return idx
#             return "default"

#         def limit(items, K):
#             """Select at most *K* elements from a list according to clip_selector."""
#             if K is None or len(items) <= K:
#                 return items
#             if self.clip_selector == "random":
#                 sel = np.random.choice(len(items), K, replace=False)
#                 sel.sort()
#                 return [items[i] for i in sel]
#             return items[:K]          # "first"

#         def pad_or_trim(v, a, T):
#             """Return (video, audio) with temporal length exactly *T*."""
#             L = v.shape[0]

#             # trim
#             if L > T:
#                 if self.clip_selector == "random":
#                     start = np.random.randint(0, L - T + 1)
#                     v, a = v[start : start + T], a[start : start + T]
#                 else:
#                     v, a = v[:T], a[:T]

#             # pad
#             elif L < T:
#                 pad = T - L
#                 if self.pad_mode == "repeat":
#                     v_pad = v[-1:].repeat(pad, *([1] * (v.ndim - 1)))
#                     a_pad = a[-1:].repeat(pad, *([1] * (a.ndim - 1)))
#                 else:  # "zeros"
#                     v_pad = torch.zeros((pad, *v.shape[1:]), dtype=v.dtype)
#                     a_pad = torch.zeros((pad, *a.shape[1:]), dtype=a.dtype)
#                 v, a = torch.cat([v, v_pad]), torch.cat([a, a_pad])

#             return v, a

#         # Construct buckets based on clip_mode
#         buckets = defaultdict(list)
#         for idx, s in enumerate(batch):
#             buckets[bucket_key(s["metadata"], idx)].append(s)

#         # Decide how many clips to keep per bucket
#         if self.clip_to == "min":
#             K = min(len(v) for v in buckets.values())
#         elif isinstance(self.clip_to, int):
#             K = self.clip_to
#         else:                      # None → no limit
#             K = None

#         if self.sample_mode == "single":
#             vid_ts, aud_ts, meta_list = [], [], []

#             for bucket in buckets.values():
#                 # Each sample already is one window
#                 kept = limit(bucket, K)
#                 if not kept:
#                     continue

#                 vids = torch.stack([s["video"] for s in kept])
#                 auds = torch.stack([s["audio"] for s in kept])
#                 metas = [s["metadata"] for s in kept]

#                 vid_ts.append(vids)
#                 aud_ts.append(auds)
#                 meta_list.extend(metas)

#             if not vid_ts: # nothing left → skip batch
#                 return None

#             videos = torch.cat(vid_ts, 0)
#             audios = torch.cat(aud_ts, 0)

#         elif self.sample_mode == "sequence":
#             selected = [s for bucket in buckets.values() for s in limit(bucket, K)]
#             if not selected:
#                 return None

#             vids, auds, meta_list, lengths = [], [], [], []
#             for s in selected:
#                 vids.append(s["video"])
#                 auds.append(s["audio"])
#                 meta_list.append(s["metadata"])
#                 lengths.append(s["video"].shape[0])

#             # Decide target temporal length
#             if self.seq_len == "min":
#                 T = min(lengths)
#             elif isinstance(self.seq_len, int):
#                 T = self.seq_len
#             else: # None → pad to longest
#                 T = max(lengths)

#             vids_pad, auds_pad = zip(*(pad_or_trim(v, a, T) for v, a in zip(vids, auds)))
#             videos = torch.stack(list(vids_pad), 0)
#             audios = torch.stack(list(auds_pad), 0)
            
#         else:
#             raise ValueError(f"Invalid sample mode: {self.sample_mode}")

#         # Organize metadata into dict of lists
#         metas = {k: [m[k] for m in meta_list] for k in meta_list[0]}
#         if self.sample_mode == "sequence":
#             metas["length"] = torch.as_tensor(lengths, dtype=torch.long)

#         # Optional ID encoding
#         if self.encode_ids:
#             ids = metas["id_source"] + metas["id_target"]
#             self.id_encoder.fit(ids)
#             metas["id_source"] = self.id_encoder.transform(metas["id_source"])
#             metas["id_target"] = self.id_encoder.transform(metas["id_target"])
#             self.id_encoder = LabelEncoder()  # reset for next batch

#         # Convert numeric lists → tensors
#         for k, v in metas.items():
#             if isinstance(v[0], (int, np.integer, numbers.Integral)):
#                 metas[k] = torch.as_tensor(v, dtype=torch.long)
#             elif isinstance(v[0], (float, np.floating, numbers.Real)):
#                 metas[k] = torch.as_tensor(v, dtype=torch.float)

#         return {"video": videos, "audio": audios, "metadata": metas}

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             collate_fn=self._collate_fn,
#             drop_last=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=self._collate_fn
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             collate_fn=self._collate_fn,
#         )

#     def teardown(self, stage=None):
#         for ds in ["train_dataset", "val_dataset", "test_dataset"]:
#             if hasattr(self, ds):
#                 getattr(self, ds).close()
                

# =============================================================================
#                SWAN-DF  +  SWAN-Idiap  Dataset  (protocol-aware)
# =============================================================================
class SWAN_DF_Dataset(Dataset):
    """
    Reads the identity protocols generated with *make_swan_df_protocols.py*.

    Parameters
    ----------
    protocol_dir : Path
        Directory that contains the **three** lists:
        ├── train_identities.txt
        ├── dev_identities.txt
        └── test_identities.txt

    split : "train" | "dev" | "test"
        Keeps samples whose `id_target` appears in the corresponding list.
    """
    _RE_REAL = re.compile(
    r"(?P<site>[1-5])_(?P<id>\d{5})_(?P<gender>[mf])_(?P<session>\d{2})_"
    r"(?P<rec>\d{2})_(?P<device>[pt])_(?P<bio>[1234])\.", re.X)

    _RE_FAKE = re.compile(
        r"(?P<prefix>[^-]+)-(?P<model>model_\d+d)-(?P<train>[^-]+)-"
        r"(?P<blend>[^-]+)-to-(?P<target>\d{5})\.", re.X)

    def __init__(
        self,
        split: Literal["train", "dev", "test"],
        root_df: str | Path,
        root_real: str | Path,
        *,
        protocol_dir: str | Path = "swan_df_protocols",
        video_processor: Optional[Callable] = None,
        audio_processor: Optional[Callable] = None,
        resolutions: Optional[set[str] | list[str]] = None,
        use_wav_audio: bool = False,
        mode: Literal["minimal", "full"] = "minimal",
    ):
        super().__init__()
        self.split = split
        self.mode  = mode
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.keep_res = None if resolutions is None else set(resolutions if isinstance(resolutions, (list, set)) else [resolutions])
        self.use_wav_audio = use_wav_audio

        # read the protocol files
        proto = Path(root_df) / protocol_dir
        train_ids = self._read_id_list(proto / "train_identities.txt")
        dev_ids   = self._read_id_list(proto / "dev_identities.txt")
        test_ids  = self._read_id_list(proto / "test_identities.txt")

        # sanity check – no leaks
        if not (train_ids.isdisjoint(dev_ids | test_ids) and dev_ids.isdisjoint(test_ids)):
            raise ValueError("train / dev / test identity lists overlap!")

        target_sets = {"train": train_ids, "dev": dev_ids, "test": test_ids}
        wanted_ids  = target_sets[split]

        # index fake and real samples
        fake_samples = self._index_swan_df(Path(root_df))
        real_samples = self._index_swan_real(Path(root_real))

        # keep only the identities we want for *this* split
        self.samples = [
            s for s in (fake_samples + real_samples)
            if s["id_target"] in wanted_ids
        ]

        if not self.samples:
            raise RuntimeError(f"No samples found for split='{split}'. "
                               "Check your protocol lists and paths.")
            
        # pre-compute per-sample weights (for dataloder balancing)
        cls_counts = {"0": 0, "1": 0}
        for s in self.samples:
            cls_counts[s["label"]] += 1
        inv_freq = {lbl: 1.0 / n for lbl, n in cls_counts.items()} # inverse frequency
        self.sample_weights = [inv_freq[s["label"]] for s in self.samples]

        # label encoders
        self.encoders = {
            "label": LabelEncoder().fit(["0", "1"]),
            "av"   : LabelEncoder().fit(["00", "11"]), # only FA-FV in SWAN-DF
        }
        
    def _read_id_list(self, path: Path) -> set[str]:
        if not path.exists():
            raise FileNotFoundError(f"Required protocol file missing: {path}")
        return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}

    def _index_swan_df(self, root: Path) -> List[Dict]:
        """
        Walk   videos/<RES>/<SRC_ID>/*.mp4   and, if present,
        audios/wav/<SRC_ID>/*.wav
        """
        # index the fake samples
        vids_root = root / "videos"
        wavs_root = root / "audios/wav"
        
        samples: List[Dict] = []
        for res_dir in vids_root.iterdir(): # 160x160 / …
            if not res_dir.is_dir():
                continue
            if self.keep_res and res_dir.name not in self.keep_res: # skip unwanted resolutions
                continue
            for src_dir in res_dir.iterdir(): # 00001 / 00002 …
                if not src_dir.is_dir():
                    continue
                src_id = src_dir.name
                for mp4 in src_dir.glob("*.mp4"):
                    m = self._RE_FAKE.search(mp4.name)
                    if not m: # safety check
                        continue
                    tgt_id = m["target"]
                    
                    # optional matching WAV
                    wav_path = None
                    if self.use_wav_audio:
                        maybe = list(
                            (wavs_root / src_id).glob(f"{mp4.stem.split('-to-')[0]}*-to-{tgt_id}.wav")
                        )
                        if maybe:
                            wav_path = str(maybe[0])
                            
                    samples.append(
                        {
                            "video_file": str(mp4),
                            "audio_file": wav_path,
                            "label": "1", "av": "11",
                            "id_source": src_id,
                            "id_target": m["target"],
                            "metadata": {
                                "video_file": str(mp4),
                                "audio_file": wav_path,
                                "resolution": res_dir.name,
                                "model"     : m["model"],
                                "train"     : m["train"],
                                "blend"     : m["blend"],
                            },
                        }
                    )
        return samples

    def _index_swan_real(self, root: Path) -> List[Dict]:
        """
        Walk SWAN-Idiap bona-fide videos only.
        Path pattern:  …/IDIAP/session_XX/(iPad|iPhone)/<ID>/*.mp4
        """
        # index the real samples
        idiap_dir = root / "IDIAP"
        
        samples: List[Dict] = []
        for sess in idiap_dir.glob("session_*"):
            for device in ("iPad", "iPhone"):
                dev_dir = sess / device
                if not dev_dir.is_dir():
                    continue
                for id_dir in dev_dir.iterdir():
                    if not id_dir.is_dir():
                        continue
                    ident = id_dir.name
                    for mp4 in id_dir.glob("*.mp4"):
                        m = self._RE_REAL.search(mp4.name)
                        if not m:
                            continue
                        bio_code = m.group("bio")
                        if bio_code not in {"2"}: # keep voice (2) | skip face (1) and eye (3)
                            continue
                        
                        samples.append(
                            {
                                "video_file": str(mp4),
                                "audio_file": None, # embedded in the video
                                "label": "0", 
                                "av": "00",
                                "id_source": ident,
                                "id_target": ident,
                                "metadata": {
                                    "video_file": str(mp4),
                                    "audio_file": None,
                                    "session": sess.name,
                                    "device" : device,
                                    "site"   : m.group("site"),
                                    "gender" : m.group("gender"),
                                    "bio"    : bio_code,
                                },
                            }
                        )
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        video, audio, info = read_video(s["video_file"])
        
        if s["audio_file"]:
            audio, sr = read_audio(s["audio_file"])
            info["audio_fps"] = sr

        if self.video_processor:
            video, vid_mask = self.video_processor(video, info["video_fps"])
        if self.audio_processor:
            audio, aud_mask = self.audio_processor(audio, info["audio_fps"])
            
        if self.video_processor and self.audio_processor:
            L = min(len(video), len(audio))
            video, audio = video[:L], audio[:L]
            if "vid_mask" in locals():
                joint = vid_mask[:L] & aud_mask[:L]
                keep = joint.nonzero(as_tuple=False).squeeze(-1)
                video, audio = video[keep], audio[keep]

        meta = s["metadata"].copy()
        meta.update(info)

        if self.mode == "minimal":
            meta = {
                "label": self.encoders["label"].transform([s["label"]])[0],
                "av"   : self.encoders["av"].transform([s["av"]])[0],
                "id_source": s["id_source"],
                "id_target": s["id_target"],
            }

        return {"video": video, "audio": audio, "metadata": meta}


# ==============================================================================
#                Generic Dataset and DataModule
# ==============================================================================
class AVDataModule(LightningDataModule):
    """
    A generic DataModule for *any* audio-video dataset class that yields
    ``{"video", "audio", "metadata"}``.

    Parameters
    ----------
    dataset_cls : Dataset subclass
        Pass either ``DeepSpeak_v1_1_Dataset`` or ``SWAN_DF_Dataset`` (or any
        other compatible class).

    dataset_kwargs : dict
        Keyword args forwarded verbatim to the dataset constructor.

    Supports sampling / padding / balancing parameters
    """

    def __init__(
        self,
        dataset_cls: Type[Dataset],
        dataset_kwargs: dict,
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        sample_mode: Literal["single", "sequence"] = "single",
        encode_ids: bool = True,
        balance_classes: bool = False,
        # BATCH-BALANCING
        clip_mode: Optional[Literal["id", "idx"]] = None,
        clip_to: None | Literal["min"] | int = None, # None == 'max'
        clip_selector: Literal["first", "random"] = "random",
        clip_selector_seed: int = 42,
        # TEMPORAL PADDING (sequence mode)
        pad_mode: Optional[Literal["repeat", "zeros"]] = "repeat",
        seq_len: None | Literal["min"] | int = None, # None == 'max'
    ):
        super().__init__()
        self.dataset_cls    = dataset_cls
        self.dataset_kwargs = dataset_kwargs
        self.balance_classes = balance_classes

        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.sample_mode  = sample_mode
        self.encode_ids   = encode_ids
        self.id_encoder   = LabelEncoder() if encode_ids else None

        # balancing / padding knobs
        self.clip_mode     = clip_mode
        self.clip_to       = clip_to
        self.clip_selector = clip_selector
        self.clip_selector_seed = clip_selector_seed
        self.pad_mode      = pad_mode
        self.seq_len       = seq_len

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            self.train_dataset = self.dataset_cls(split="train", **self.dataset_kwargs)
            self.val_dataset   = self.dataset_cls(split="dev",   **self.dataset_kwargs)
                
        if stage in ("test", None):
            self.test_dataset  = self.dataset_cls(split="test",  **self.dataset_kwargs)

    def _collate_fn(self, batch):
        """
        Balancing logic
        ───────────────
        1.  Groups are formed with 'clip_mode' key:
            - clip_mode="id"  →  identity  (meta["id_source"])
            - clip_mode="idx" →  each item by its original sample idx
            - otherwise       →  single bucket ("default")
        2.  For every group we keep at most *K* sequences, where *K* is defined by 'clip_to':
            - clip_to="min"  →  K = min(group sizes in the batch)
            - clip_to=<int>  →  K = clip_to
            - clip_to=None   →  keep everything
        3.  If a group has more than *K* clips, the clips are selected
            according to `clip_selector`:
            - clip_selector="first"  →  first *K* clips
            - clip_selector="random" →  random *K* clips
        4.  (Sequence mode) The remaining clips are then trimmed or padded in time to length L, so they can be stacked into one tensor. Controled by 'pad_mode':
            - pad_mode="repeat"  →  repeat sequence to fill the time dimension
            - pad_mode="zeros"   →  pad with zeros to fill the time dimension
        and 'seq_len':
            - seq_len="min"  →  L = min(len of sequences in the batch)
            - seq_len=<int>  →  L = seq_len
            - seq_len=None   →  keep full length of all sequences
        """
        rng = default_rng(self.clip_selector_seed)

        def bucket_key(meta, idx):
            if self.clip_mode == "id":
                return meta["id_source"]
            if self.clip_mode == "idx":
                return idx
            return "default"

        def limit_bucket(items, k):
            """Select at most *K* elements from a list according to clip_selector."""
            if k is None or len(items) <= k:
                return items
            if self.clip_selector == "random":
                idx = rng.choice(len(items), k, replace=False)
                idx.sort()
                return [items[i] for i in idx]
            return items[:k]  # first

        def pad_or_trim(v, a, target_len):
            """Return (video, audio) clipped or padded to *target_len* frames."""
            cur_len = v.shape[0]
            
            # trim
            if cur_len > target_len:
                if self.clip_selector == "random":
                    start = rng.choice(0, cur_len - target_len + 1)
                    v, a = v[start : start + target_len], a[start : start + target_len]
                else:
                    v, a = v[:target_len], a[:target_len]
            # pad
            elif cur_len < target_len:
                pad_len = target_len - cur_len
                if self.pad_mode == "repeat":
                    v_pad = v[-1:].repeat(pad_len, *([1] * (v.ndim - 1)))
                    a_pad = a[-1:].repeat(pad_len, *([1] * (a.ndim - 1)))
                else:
                    v_pad = torch.zeros((pad_len, *v.shape[1:]), dtype=v.dtype)
                    a_pad = torch.zeros((pad_len, *a.shape[1:]), dtype=a.dtype)
                v, a = torch.cat([v, v_pad]), torch.cat([a, a_pad])
            return v, a

        # Construct buckets based on clip_mode
        buckets = defaultdict(list)
        for idx, sample in enumerate(batch):
            buckets[bucket_key(sample["metadata"], idx)].append(sample)

        # Decide how many clips to keep per bucket (K)
        if self.clip_to == "min":
            K = min(len(v) for v in buckets.values())
        elif isinstance(self.clip_to, int):
            K = self.clip_to
        else:
            K = None   # keep all

        # Sample from the buckets
        if self.sample_mode == "single":
            vids, auds, metas = [], [], []
            for items in buckets.values():
                # flatten windows inside every sample first
                windows = [
                    (vw, aw, meta)
                    for s in limit_bucket(items, K)
                    for vw, aw, meta in zip(s["video"], s["audio"], [s["metadata"]]*len(s["video"]))
                ]
                
                # Select windows from buckets
                if self.clip_selector == "random" and K and len(windows) > K:
                    idx = rng.choice(len(windows), K, replace=False)
                    idx.sort()
                    windows = [windows[i] for i in idx]
                else:
                    windows = windows[:K] if K else windows
                    
                # Stack windows
                if windows:
                    vids.append(torch.stack([w[0] for w in windows]))
                    auds.append(torch.stack([w[1] for w in windows]))
                    metas.extend(w[2] for w in windows)

            if not vids:  # all buckets empty -> skip this batch
                return None

            video = torch.cat(vids, 0)
            audio = torch.cat(auds, 0)

        elif self.sample_mode == "sequence":
            
            # Select clips from buckets
            chosen = [s for b in buckets.values() for s in limit_bucket(b, K)]
            if not chosen:
                return None
            
            # Collect video/audio/metadata
            vids, auds, metas, lens = [], [], [], []
            for s in chosen:
                vids.append(s["video"])
                auds.append(s["audio"])
                metas.append(s["metadata"])
                lens.append(s["video"].shape[0]) # number of windows
                
            # Determine target temporal length
            if self.seq_len == "min":
                L = min(lens)
            elif isinstance(self.seq_len, int):
                L = self.seq_len
            else:  # seq_len is None → pad to longest
                L = max(lens)
                
            # Pad or trim sequences
            vids, auds = zip(*(pad_or_trim(v, a, L) for v, a in zip(vids, auds)))
            video = torch.stack(list(vids))
            audio = torch.stack(list(auds))
        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")

        # Organize metadata into dict of lists
        meta_dict = {k: [m[k] for m in metas] for k in metas[0]}
        
        if self.sample_mode == "sequence":
            meta_dict["length"] = torch.as_tensor(lens, dtype=torch.long)

        # Optional label-encoding of identities
        if self.encode_ids:
            all_ids = meta_dict["id_source"] + meta_dict["id_target"]
            self.id_encoder.fit(all_ids)
            meta_dict["id_source"] = self.id_encoder.transform(meta_dict["id_source"])
            meta_dict["id_target"] = self.id_encoder.transform(meta_dict["id_target"])

        # Tensorise simple numeric metadata
        for k, v in meta_dict.items():
            if isinstance(v[0], (int, np.integer, numbers.Integral)):
                meta_dict[k] = torch.as_tensor(v, dtype=torch.long)
            elif isinstance(v[0], (float, np.floating, numbers.Real)):
                meta_dict[k] = torch.as_tensor(v, dtype=torch.float)

        return {"video": video, "audio": audio, "metadata": meta_dict}

    def train_dataloader(self):
        if self.balance_classes:
            sampler = WeightedRandomSampler(
                weights=self.train_dataset.sample_weights,
                num_samples=len(self.train_dataset.sample_weights),
                replacement=True,
            )
        else:
            sampler = None
        
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True if not sampler else False, num_workers=self.num_workers,
                          sampler=sampler,
                          collate_fn=self._collate_fn, persistent_workers=self.num_workers > 0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self._collate_fn,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self._collate_fn,
                          persistent_workers=self.num_workers > 0)


class PreparedAVH5Dataset(Dataset):
    """
    Generic loader for *any* H5 + flat-index pair produced by perprocess_dataset.py script (DeepSpeak, SWAN-DF, …).
    """
    def __init__(
        self,
        split: Literal["train", "dev", "test"],
        data_dir: str | Path,
        sample_mode: Literal["single", "sequence"] = "single",
        *,
        av_codes: List[str] = ["00", "01", "10", "11"],
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_mode = sample_mode

        self.h5_path   = self.data_dir / f"{split}.h5"
        self.index_path = self.data_dir / f"{split}_flat_index.json"

        self._prepare_label_encoders(av_codes)
        self._load_index()

    def _prepare_label_encoders(self, av_codes: Optional[List[str]]):
        self.encoders: Dict[str, LabelEncoder] = {
            "label": LabelEncoder().fit(["0", "1"]),
            "av":    LabelEncoder().fit(av_codes),
        }

    def _load_index(self):
        self.h5_file = h5py.File(self.h5_path, "r")

        if self.sample_mode == "single":
            with open(self.index_path) as f:
                self.flat_index = json.load(f)
        else:  # sequence
            self.video_ids = list(self.h5_file.keys())
            
        # build per-sample weights
        labels = []
        if self.sample_mode == "single":
            # look up label once per *window* (cheap: just read group attr)
            for ent in self.flat_index:
                meta = json.loads(
                    self.h5_file[ent["sample_id"]].attrs["metadata"]
                )
                labels.append(meta["av"])
        else:               # sequence mode – one label per clip
            for vid in self.video_ids:
                meta = json.loads(self.h5_file[vid].attrs["metadata"])
                labels.append(meta["av"])
                
        # label counts and inverse-frequency weights
        counts = Counter(labels)
        inv_freq = {k: 1.0 / v for k, v in counts.items()}
        self.sample_weights = [inv_freq[str(lbl)] for lbl in labels]

    def __len__(self):
        return len(self.flat_index) if self.sample_mode == "single" else len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.sample_mode == "single":
            ent = self.flat_index[idx]
            vid  = ent["sample_id"]
            widx = ent["window_idx"]
            video = torch.as_tensor(self.h5_file[vid]["video"][widx])
            audio = torch.as_tensor(self.h5_file[vid]["audio"][widx])
            meta  = json.loads(self.h5_file[vid].attrs["metadata"])
        else: # sequence
            vid   = self.video_ids[idx]
            video = torch.as_tensor(self.h5_file[vid]["video"][:])
            audio = torch.as_tensor(self.h5_file[vid]["audio"][:])
            meta  = json.loads(self.h5_file[vid].attrs["metadata"])

        # encode labels
        meta["label"] = self.encoders["label"].transform([meta["label"]])[0]
        meta["av"]    = self.encoders["av"].transform([meta["av"]])[0]

        return {"video": video, "audio": audio, "metadata": meta}

    def close(self):
        self.h5_file.close()
        
        
class PreparedAVDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        sample_mode: Literal["single", "sequence"] = "single",
        encode_ids: bool = True,
        clip_mode: Optional[Literal["id", "idx"]] = None,
        clip_to: Literal["min"] | int = "min",
        clip_selector: Literal["first", "random"] = "random",
        pad_mode: Optional[Literal["repeat", "zeros"]] = "repeat",
        seq_len: None | Literal["min"] | int = None,
        dataset_kwargs: dict | None = None,
        *,
        balance_classes: bool = False,
    ):
        super().__init__()
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.sample_mode  = sample_mode
        self.encode_ids   = encode_ids
        self.clip_mode    = clip_mode
        self.clip_to      = clip_to
        self.clip_selector = clip_selector
        self.pad_mode     = pad_mode
        self.seq_len      = seq_len
        self.dataset_kwargs = dataset_kwargs or {}
        self.balance_classes = balance_classes

        if self.encode_ids:
            self.id_encoder = LabelEncoder()

    def setup(self, stage: str | None = None):
        if stage in ("fit", None):
            self.train_dataset = PreparedAVH5Dataset("train", **self.dataset_kwargs)
            self.val_dataset   = PreparedAVH5Dataset("dev",   **self.dataset_kwargs)
        if stage in ("test", None):
            self.test_dataset  = PreparedAVH5Dataset("test",  **self.dataset_kwargs)

    def _collate_fn(self, batch):
        def bucket_key(meta, idx):
            if self.clip_mode == "id":
                return meta["id_source"]
            if self.clip_mode == "idx":
                return idx
            return "default"

        def limit(items, K):
            """Select at most *K* elements from a list according to clip_selector."""
            if K is None or len(items) <= K:
                return items
            if self.clip_selector == "random":
                sel = np.random.choice(len(items), K, replace=False)
                sel.sort()
                return [items[i] for i in sel]
            return items[:K]          # "first"

        def pad_or_trim(v, a, T):
            """Return (video, audio) with temporal length exactly *T*."""
            L = v.shape[0]

            # trim
            if L > T:
                if self.clip_selector == "random":
                    start = np.random.randint(0, L - T + 1)
                    v, a = v[start : start + T], a[start : start + T]
                else:
                    v, a = v[:T], a[:T]

            # pad
            elif L < T:
                pad = T - L
                if self.pad_mode == "repeat":
                    v_pad = v[-1:].repeat(pad, *([1] * (v.ndim - 1)))
                    a_pad = a[-1:].repeat(pad, *([1] * (a.ndim - 1)))
                else:  # "zeros"
                    v_pad = torch.zeros((pad, *v.shape[1:]), dtype=v.dtype)
                    a_pad = torch.zeros((pad, *a.shape[1:]), dtype=a.dtype)
                v, a = torch.cat([v, v_pad]), torch.cat([a, a_pad])

            return v, a

        # Construct buckets based on clip_mode
        buckets = defaultdict(list)
        for idx, s in enumerate(batch):
            buckets[bucket_key(s["metadata"], idx)].append(s)

        # Decide how many clips to keep per bucket
        if self.clip_to == "min":
            K = min(len(v) for v in buckets.values())
        elif isinstance(self.clip_to, int):
            K = self.clip_to
        else:                      # None → no limit
            K = None

        if self.sample_mode == "single":
            vid_ts, aud_ts, meta_list = [], [], []

            for bucket in buckets.values():
                # Each sample already is one window
                kept = limit(bucket, K)
                if not kept:
                    continue

                vids = torch.stack([s["video"] for s in kept])
                auds = torch.stack([s["audio"] for s in kept])
                metas = [s["metadata"] for s in kept]

                vid_ts.append(vids)
                aud_ts.append(auds)
                meta_list.extend(metas)

            if not vid_ts: # nothing left → skip batch
                return None

            videos = torch.cat(vid_ts, 0)
            audios = torch.cat(aud_ts, 0)

        elif self.sample_mode == "sequence":
            selected = [s for bucket in buckets.values() for s in limit(bucket, K)]
            if not selected:
                return None

            vids, auds, meta_list, lengths = [], [], [], []
            for s in selected:
                vids.append(s["video"])
                auds.append(s["audio"])
                meta_list.append(s["metadata"])
                lengths.append(s["video"].shape[0])

            # Decide target temporal length
            if self.seq_len == "min":
                T = min(lengths)
            elif isinstance(self.seq_len, int):
                T = self.seq_len
            else: # None → pad to longest
                T = max(lengths)

            vids_pad, auds_pad = zip(*(pad_or_trim(v, a, T) for v, a in zip(vids, auds)))
            videos = torch.stack(list(vids_pad), 0)
            audios = torch.stack(list(auds_pad), 0)
            
        else:
            raise ValueError(f"Invalid sample mode: {self.sample_mode}")

        # Organize metadata into dict of lists
        metas = {k: [m[k] for m in meta_list] for k in meta_list[0]}
        if self.sample_mode == "sequence":
            metas["length"] = torch.as_tensor(lengths, dtype=torch.long)

        # Optional ID encoding
        if self.encode_ids:
            ids = metas["id_source"] + metas["id_target"]
            self.id_encoder.fit(ids)
            metas["id_source"] = self.id_encoder.transform(metas["id_source"])
            metas["id_target"] = self.id_encoder.transform(metas["id_target"])
            self.id_encoder = LabelEncoder()  # reset for next batch

        # Convert numeric lists → tensors
        for k, v in metas.items():
            if isinstance(v[0], (int, np.integer, numbers.Integral)):
                metas[k] = torch.as_tensor(v, dtype=torch.long)
            elif isinstance(v[0], (float, np.floating, numbers.Real)):
                metas[k] = torch.as_tensor(v, dtype=torch.float)

        return {"video": videos, "audio": audios, "metadata": metas}

    def train_dataloader(self):
        if self.balance_classes and hasattr(self.train_dataset, "sample_weights"):
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(self.train_dataset.sample_weights,
                                            num_samples=len(self.train_dataset),
                                            replacement=True)
        else:
            sampler = None
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler, # sampler OR plain shuffle
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self._collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self._collate_fn)

    def teardown(self, stage=None):
        for attr in ("train_dataset", "val_dataset", "test_dataset"):
            if hasattr(self, attr):
                getattr(self, attr).close()

# =============================================================================
DATASET_MAP = {
    "DeepSpeak_v1_1": {
        "original": DeepSpeak_v1_1_Dataset,
        "preprocessed": PreparedAVH5Dataset,
    },
    "SWAN_DF": {
        "original": SWAN_DF_Dataset,
        "preprocessed": PreparedAVH5Dataset,
    },
}

DatasetType = Literal["DeepSpeak_v1_1", "SWAN_DF"]


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
    "DeepSpeak_v1_1": {
        "original": AVDataModule,
        "preprocessed": PreparedAVDataModule,
    },
    "SWAN_DF": {
        "original": AVDataModule,
        "preprocessed": PreparedAVDataModule,
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
