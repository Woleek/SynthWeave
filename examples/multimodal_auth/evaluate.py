import argparse
from typing import Dict
import torch
from src.pipe import MultiModalAuthPipeline, ImagePreprocessor, AudioPreprocessor, AdaFace, ReDimNet
from synthweave.utils.datasets import get_datamodule
from synthweave.utils.fusion import get_fusion
from pathlib import Path
import json
from tqdm.auto import tqdm
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AveragePrecision, 
    AUROC
)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="encoded_data/DeepSpeak_v1_1",
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--fusion", 
        type=str,
        required=True,
        help="Fusion method to evaluate.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["binary", "fine-grained"],
        help="Task type.",
    )

    return parser.parse_args()

def setup(args: argparse.Namespace):
    
    ds_kwargs = {
        "data_dir": args.data_dir,
        "preprocessed": True,
        "sample_mode": "sequence",
    }

    dm = get_datamodule(
        "DeepSpeak_v1_1",
        batch_size=1, # NOTE: currently single window fusions don't ignore padding
        dataset_kwargs=ds_kwargs,
        sample_mode="sequence",
        # clip_mode = None,
        # pad_mode = 'zeros'
    )

    dm.setup()

    FUSION = args.fusion
    TASK = args.task

    path = Path("logs") / TASK / FUSION
    path = sorted(path.glob("version_*"))[-1]
    
    print(f"Loading model: '{FUSION}' for task '{TASK}' from {path}")

    # config
    model_args = json.loads((path / "args.json").read_text())
    model_args = dotdict(model_args)

    # best checkpoint
    ckpt = path / "checkpoints"
    ckpt = sorted(ckpt.glob("epoch=*.ckpt"))[-1]


    models = {"audio": torch.nn.Identity(), "video": torch.nn.Identity()}
    EMB_DIM = model_args.emb_dim

    fusion = get_fusion(
        fusion_name=FUSION,
        output_dim=EMB_DIM,
        modality_keys=["video", "audio"],
        out_proj_dim=model_args.proj_dim,
        num_att_heads=4,  # only for attention-based fusions
        dropout=model_args.dropout,
    )

    if TASK == "binary":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, 1), torch.nn.Sigmoid()
        )
    elif TASK == "fine-grained":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, 4), torch.nn.Softmax(dim=1)
        )

    pipe = MultiModalAuthPipeline(
        models=models,
        fusion=fusion,
        detection_head=detection_head,
        freeze_backbone=True,
    )

    state_dict = torch.load(ckpt, map_location="cpu")['state_dict']
    state_dict = {k.replace("pipeline.", ""): v for k, v in state_dict.items()}
    pipe.load_state_dict(state_dict, strict=False)

    pipe = pipe.to(device)
    pipe.eval();

    return dm, pipe, path

def get_metrics(task):
    metrics: Dict[str, Dict[str, torch.nn.Module]] = {}
    metric_kwargs = (
        {"task": "binary"}
        if task == "binary"
        else {"task": "multiclass", "num_classes": 4, "average": "macro"}
    )

    base = dict(
        acc=Accuracy(**metric_kwargs),
        prec=Precision(**metric_kwargs),
        rec=Recall(**metric_kwargs),
        f1=F1Score(**metric_kwargs),
        ap=AveragePrecision(**metric_kwargs),
        auc=AUROC(**metric_kwargs),
    )

    for split in ("train", "val", "test"):
        metrics[split] = {k: v.clone() for k, v in base.items()}
        
    for split_metrics in metrics.values():
        for metric in split_metrics.values():
            metric.to(device)
        
    return metrics

def _update_metrics(metrics, probs, preds, labels, split):
    if preds.ndim == 0:
        preds = preds.unsqueeze(0)
    if probs.ndim == 0:
        probs = probs.unsqueeze(0)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
        
    for k, metric in metrics[split].items():
        if k in {"ap", "auc"}:
            metric.update(probs, labels)
        else:
            metric.update(preds, labels)
            
def _eval_split(loader, pipe, metrics, split, task):
    for sample in tqdm(loader, desc=split):
        sample["video"] = sample["video"].squeeze(0).to(device)
        sample["audio"] = sample["audio"].squeeze(0).to(device)

        with torch.no_grad():
            out = pipe(sample)
            
            probs = out["logits"]
            
            # Average logits (soft aggregation)
            if task == "binary":
                final_prob = torch.mean(probs)
                final_pred = (final_prob > 0.5).long()
            else:
                final_prob = torch.mean(probs, dim=0)
                final_pred = torch.argmax(final_prob)
            
            gt = sample['metadata']["label"].to(device)
            
            _update_metrics(metrics, final_prob, final_pred, gt, split)
            
def save_metrics(metrics, path):
    for split, split_metrics in metrics.items():
        for k, v in split_metrics.items():
            v = v.compute()
            v = v.item()
            metrics[split][k] = v

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def evaluate(dm, pipe, metrics):
    
    # TRAIN SET
    train_loader = dm.train_dataloader()
    _eval_split(train_loader, pipe, metrics, "train", args.task)
    print("Train Set Metrics")
    for k, v in metrics["train"].items():
        print(f"{k}: {v.compute().item(): .3f}")
        
    print("\n" + "-"*50 + "\n")
        
    # DEV SET
    val_loader = dm.val_dataloader()
    _eval_split(val_loader, pipe, metrics, "val", args.task)
    print("Validation Set Metrics")
    for k, v in metrics["val"].items():
        print(f"{k}: {v.compute().item(): .3f}")
        
    print("\n" + "-"*50 + "\n")
        
    # TEST SET
    test_loader = dm.test_dataloader()
    _eval_split(test_loader, pipe, metrics, "test", args.task)
    print("Test Set Metrics")
    for k, v in metrics["test"].items():
        print(f"{k}: {v.compute().item(): .3f}")
        
        
if __name__ == "__main__":
    args = parse_args()
    dm, pipe, path = setup(args)
    
    metrics = get_metrics(args.task)
    
    evaluate(dm, pipe, metrics)
    
    save_metrics(metrics, path / "results.json")