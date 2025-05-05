import argparse
import json
import os
import re
import torch

from pathlib import Path
from unimodal import AdaFace, ReDimNet
from pipe import MultiModalAuthPipeline, ImagePreprocessor, AudioPreprocessor
from synthweave.utils.datasets import get_datamodule
from synthweave.utils.fusion import get_fusion
from synthweave.fusion import FusionType
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer

from train import DeepfakeDetector

torch.set_float32_matmul_precision("medium")

FUSION = input("Fusion name: ").strip().upper()

CHECKPOINT_ROOT = Path("./logs/binary")
checkpoint = [*(CHECKPOINT_ROOT / FUSION).rglob("**/epoch=*.ckpt")]
# sort by version
checkpoint.sort(key=lambda x: int(re.search(r"version_(\d+)", str(x)).group(1)))
if len(checkpoint) == 1:
    checkpoint = checkpoint[0]
else:
    version = input(f"Version number (Aval: {list(range(len(checkpoint)))}): ").strip()
    checkpoint = checkpoint[int(version)]


def load_saved_hyperparams(ckpt_path: str) -> dict:
    """Find `args.json` three levels above the checkpoint file."""
    version_dir = os.path.abspath(os.path.join(ckpt_path, os.pardir, os.pardir))
    args_file = os.path.join(version_dir, "args.json")
    if not os.path.isfile(args_file):
        raise FileNotFoundError(f"Could not locate {args_file}")
    with open(args_file) as f:
        return json.load(f)


saved_args = load_saved_hyperparams(checkpoint)


def build_pipeline(cfg):
    emb_dim = cfg["emb_dim"]

    # Backbones
    if cfg["encoded"]:
        models = {"audio": torch.nn.Identity(), "video": torch.nn.Identity()}
    else:
        models = {
            "audio": ReDimNet(freeze=True),
            "video": AdaFace(path="../../../models/", freeze=True),
        }

    # Fusion + detection head
    fusion = get_fusion(
        fusion_name=cfg["fusion"],
        output_dim=emb_dim,
        modality_keys=["video", "audio"],
        out_proj_dim=cfg["proj_dim"],
    )
    if cfg["task"] == "binary":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 1), torch.nn.Sigmoid()
        )
    else:  # fine-grained (4-way)
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4), torch.nn.Softmax(dim=1)
        )

    return MultiModalAuthPipeline(
        models=models,
        fusion=fusion,
        detection_head=detection_head,
        freeze_backbone=True,
    )


pipeline = build_pipeline(saved_args)
model = DeepfakeDetector(
    pipeline=pipeline, detection_task=saved_args["task"], lr=saved_args["lr"]
)


def build_datamodule(cfg):
    if cfg["preprocessed"] or cfg["encoded"]:
        ds_kwargs = {
            "data_dir": cfg["data_dir"],
            "preprocessed": True,
            "sample_mode": "single",
        }
    else:
        vid_proc = ImagePreprocessor(
            window_len=cfg["window_len"],
            step=cfg["window_step"],
            head_pose_dir="../../../models/head_pose",
        )
        aud_proc = AudioPreprocessor(
            window_len=cfg["window_len"], step=cfg["window_step"]
        )
        ds_kwargs = {
            "video_processor": vid_proc,
            "audio_processor": aud_proc,
            "mode": "minimal",
            "sample_mode": "single",
        }

    return get_datamodule(
        "DeepSpeak_v1_1",
        batch_size=cfg["batch_size"],
        dataset_kwargs=ds_kwargs,
        sample_mode="single",
        clip_mode=None,
        clip_to=None,
        clip_selector=None,
        num_workers=4,
    )


dm = build_datamodule(saved_args)
dm.setup()


trainer = Trainer(
    accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1, logger=False
)


dataloaders = [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]
all_results = {}

print("▶  Running evaluation on all splits...")

for dl in dataloaders:
    split = dl.dataset.split
    print(f"→ Evaluating split: {split}")

    # Run
    results = trainer.test(model=model, dataloaders=dl, ckpt_path=checkpoint)

    results = {key.split("/")[-1]: val for key, val in results[0].items()}

    all_results[split] = results
print("✔️  Evaluation done.")


results_json_path = os.path.join(checkpoint.parent.parent, f"results.json")
with open(results_json_path, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"✔️ Results saved to: {results_json_path}")
