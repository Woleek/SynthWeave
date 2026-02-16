import argparse
import os
from typing import Dict
import torch
from src.pipe import (
    MultiModalAuthPipeline,
)
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
    AUROC,
)
from torchmetrics.functional import f1_score
from torchmetrics import ConfusionMatrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib
from train import ClassifierHead
from sklearn.metrics import roc_curve, f1_score
import time

matplotlib.use("Agg")


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_best_checkpoint(ckpt_dir: Path) -> Path:
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        try:
            state = torch.load(last_ckpt, map_location="cpu", weights_only=False)
            callbacks = state.get("callbacks", {})
            for callback_state in callbacks.values():
                if isinstance(callback_state, dict) and "best_model_path" in callback_state:
                    best_path = callback_state.get("best_model_path")
                    if best_path:
                        best_path = Path(best_path)
                        if best_path.exists():
                            return best_path
                        candidate = ckpt_dir / best_path.name
                        if candidate.exists():
                            return candidate
        except Exception:
            pass

    ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No epoch checkpoints found in: {ckpt_dir}")
    return ckpts[-1]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["DeepSpeak_v1_1", "SWAN_DF"],
        required=True,
        help="Which dataset to load",
    )
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
    parser.add_argument(
        "--trained_on",
        choices=["DeepSpeak_v1_1", "SWAN_DF"],
        required=True,
        help="Which dataset the model was trained on",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Optional cap for number of validation recordings to evaluate.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Optional cap for number of test recordings to evaluate.",
    )

    return parser.parse_args()


def setup(args: argparse.Namespace):

    ds_kwargs = {
        "data_dir": args.data_dir,
        "preprocessed": True,
        "sample_mode": "sequence",
    }

    dm = get_datamodule(
        args.dataset,
        batch_size=1,
        dataset_kwargs=ds_kwargs,
        sample_mode="sequence",
        num_workers=max(1, (os.cpu_count() or 1) // 2),
    )

    dm.setup()

    FUSION = args.fusion
    TASK = args.task
    TRAIN_DATASET = args.trained_on

    path = Path("logs") / TRAIN_DATASET / TASK / FUSION
    versions = sorted(path.glob("version_*"), key=lambda x: x.stat().st_ctime)
    if not versions:
        raise FileNotFoundError(f"No model versions found in: {path}")
    path = versions[-1]

    print(f"Loading model: '{FUSION}' for task '{TASK}' from {path}")

    # config
    model_args = json.loads((path / "args.json").read_text())
    model_args = dotdict(model_args)

    # best checkpoint
    ckpt_dir = path / "checkpoints"
    ckpt = _resolve_best_checkpoint(ckpt_dir)
    print(f"Using checkpoint: {ckpt}")

    models = {"audio": torch.nn.Identity(), "video": torch.nn.Identity()}
    EMB_DIM = model_args.emb_dim

    fusion = get_fusion(
        fusion_name=FUSION,
        output_dim=EMB_DIM,
        modality_keys=["video", "audio"],
        input_dims={
            "video": 512,
            "audio": 192,
        },
        out_proj_dim=model_args.proj_dim,
        num_att_heads=2,
        n_layers=2,
        dropout=model_args.dropout,
    )

    if args.task == "binary":
        detection_head = ClassifierHead(input_dim=EMB_DIM, num_classes=1)
        pass
    elif args.task == "fine-grained":
        assert (
            args.dataset != "SWAN_DF"
        ), "Fine-grained task is not supported for SWAN_DF, as it has only RA-RV and FA-FV classes."
        detection_head = ClassifierHead(input_dim=EMB_DIM, num_classes=4)

    pipe = MultiModalAuthPipeline(
        models=models,
        fusion=fusion,
        detection_head=detection_head,
        freeze_backbone=True,
    )

    state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    state_dict = {k.replace("pipeline.", ""): v for k, v in state_dict.items()}
    pipe.load_state_dict(state_dict, strict=False)

    pipe = pipe.to(device)
    pipe.eval()

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

    for split in ("val", "test"):
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


AV_CLASSES = ["RA-RV", "RA-FV", "FA-RV", "FA-FV"]
AV_COLORS = plt.cm.tab10.colors[:4]


def plot_tsne(embeddings, labels, title, save_path):
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    if len(embeddings) == 0 or len(embeddings) != len(labels):
        print(
            f"[warn] Skipping plot: no data or mismatch (embeddings={len(embeddings)}, labels={len(labels)})"
        )
        return

    if len(np.unique(labels)) > len(AV_CLASSES):
        print("[warn] Found label outside expected 0-3 range")

    if len(embeddings) < 2:
        print("[warn] Skipping plot: t-SNE needs at least 2 samples")
        return

    perplexity = min(30, len(embeddings) - 1)
    if perplexity < 1:
        print("[warn] Skipping plot: invalid t-SNE perplexity")
        return

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot each class separately
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (cls_name, cls_color) in enumerate(zip(AV_CLASSES, AV_COLORS)):
        mask = labels == i
        if np.any(mask):
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=cls_name,
                color=cls_color,
                s=18,
                alpha=0.8,
            )

    ax.set_title(title)
    ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confmat(cm: np.ndarray, title: str, save_path: Path):
    """
    Plot and save a confusion matrix (2×2 or 4×4). 
    Places the x‐axis tick labels (predicted classes) on the bottom,
    and ensures y‐axis tick labels (true classes) are visible on the left.
    """
    n_classes = cm.shape[0]

    # Decide which set of labels to use
    if n_classes == 2:
        class_names = ["Negative", "Positive"]
    elif n_classes == 4:
        class_names = AV_CLASSES
    else:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=ax)

    # Move the x‐axis ticks (and labels) to the bottom
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")

    # Set title and axis labels
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("True", labelpad=8)

    # Set tick locations and tick labels for both axes
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)

    # Annotate each cell with the count
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}", va="center", ha="center", color=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    
def plot_av_pred_matrix(av_pred_counts: np.ndarray, title: str, save_path: Path):
    """
    Plot and save a 4 × 2 matrix whose rows are AV_CLASSES and columns are [Real, Fake].
    av_pred_counts should be a NumPy array of shape (4, 2), where
      av_pred_counts[i, j] = # of samples whose AV class = i and whose predicted binary = j.
    """
    n_av, n_pred = av_pred_counts.shape
    assert n_av == len(AV_CLASSES) and n_pred == 2, \
        f"Expected shape (4,2), got {av_pred_counts.shape}"

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(av_pred_counts, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("AV Ground Truth")

    # Set ticks: 4 rows (AV), 2 cols (Neg/Pos)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_yticks(np.arange(n_av))
    ax.set_yticklabels(AV_CLASSES)

    # Annotate counts
    thresh = av_pred_counts.max() / 2.0
    for i in range(n_av):
        for j in range(n_pred):
            color = "white" if av_pred_counts[i, j] > thresh else "black"
            ax.text(j, i, f"{av_pred_counts[i, j]}", va="center", ha="center", color=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def calculate_eer(labels, probs):
    """Calculate Equal Error Rate (EER) and optimal threshold"""
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    if np.unique(labels).size < 2:
        print("[warn] EER undefined for single-class labels; using fallback threshold=0.50")
        return float("nan"), 0.5

    fpr, tpr, thresholds = roc_curve(labels, probs)
    if len(thresholds) == 0:
        print("[warn] Empty ROC thresholds; using fallback threshold=0.50")
        return float("nan"), 0.5

    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    return eer, eer_threshold


def best_f1_threshold(labels, probs, num_steps: int = 1001):
    thresholds = np.linspace(0.0, 1.0, num_steps)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(np.int64)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_f1, best_thr


def _eval_split(loader, pipe, metrics, split, task, save_path, max_samples: int | None = None):
    probs, gts = [], []
    audio_emb, video_emb, fused_emb = [], [], []
    av_labels = []
    total_inference_time = 0.0
    n_samples = 0

    cm = (
        ConfusionMatrix(task="binary", num_classes=2).to(device)
        if task == "binary"
        else ConfusionMatrix(task="multiclass", num_classes=4).to(device)
    )

    for idx, sample in enumerate(tqdm(loader, desc=split)):
        if max_samples is not None and idx >= max_samples:
            break

        sample["video"] = sample["video"].squeeze(0).to(device)
        sample["audio"] = sample["audio"].squeeze(0).to(device)

        with torch.no_grad():
            start_time = time.time()
            out = pipe(sample)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            n_samples += 1
            logits = out["logits"].squeeze(-1)

            if task == "binary":
                win_prob = torch.sigmoid(logits)
                clip_prob = win_prob.mean()
            else:
                win_prob = torch.softmax(logits, dim=-1)
                clip_prob = win_prob.mean(0)

            gt_key = "label" if task == "binary" else "av"
            gt = sample["metadata"][gt_key][0].to(device)

            probs.append(clip_prob)
            gts.append(gt)

            if {"audio_proj", "video_proj", "embedding"} <= out.keys():
                audio_emb.append(out["audio_proj"].mean(0).cpu().numpy())
                video_emb.append(out["video_proj"].mean(0).cpu().numpy())
                fused_emb.append(out["embedding"].mean(0).cpu().numpy())
                av_labels.append(sample["metadata"]["av"].item())

    if len(probs) == 0:
        raise RuntimeError(
            f"No samples evaluated for split='{split}'. Check loader/data or increase max_samples."
        )

    probs_tensor = torch.stack(probs).to(device)
    gts_tensor = torch.stack(gts).to(device)

    if task == "binary":
        if split == "val":
            val_probs_np = probs_tensor.cpu().numpy()
            val_gts_np = gts_tensor.cpu().numpy()
            eer, eer_thr = calculate_eer(val_gts_np, val_probs_np)
            best_f1, f1_thr = best_f1_threshold(val_gts_np, val_probs_np)
            best_thr = eer_thr
            _eval_split.best_thr = best_thr
            print(f"[VAL] EER={eer:.4f} @ thr_eer={eer_thr:.2f}")
            print(f"[VAL] Best F1={best_f1:.4f} @ thr_f1={f1_thr:.2f}")
            print(f"[VAL] Selected threshold (EER calibration)={best_thr:.2f}")
            _eval_split.val_eer = eer
            _eval_split.val_best_f1 = best_f1
            _eval_split.val_threshold = float(best_thr)
        else:
            best_thr = getattr(_eval_split, "best_thr", 0.5)
            print(f"[TEST] Using threshold from val: {best_thr:.2f}")
            test_probs_np = probs_tensor.cpu().numpy()
            test_gts_np = gts_tensor.cpu().numpy()
            test_eer, _ = calculate_eer(test_gts_np, test_probs_np)
            _eval_split.test_eer = test_eer
            print(f"[TEST] EER: {test_eer:.4f}")

        preds = (probs_tensor >= best_thr).long()
    else:
        preds = probs_tensor.argmax(dim=1)
        
    if task == "binary":
        av_array = np.array(av_labels)
        preds_np = preds.cpu().numpy()

        av_pred_counts = np.zeros((4, 2), dtype=int)

        for i in range(len(preds_np)):
            av_idx = int(av_array[i])
            pred_label = int(preds_np[i])
            av_pred_counts[av_idx, pred_label] += 1

        plot_av_pred_matrix(
            av_pred_counts,
            f"{split.title()} set – AV vs. Binary‐Prediction",
            save_path / f"{split}_av_vs_pred_matrix.png",
        )

        print(f"[{split.upper()}] AV→Prediction breakdown:")
        for idx, cls_name in enumerate(AV_CLASSES):
            neg_count = av_pred_counts[idx, 0]
            pos_count = av_pred_counts[idx, 1]
            print(f"    {cls_name}:  Neg={neg_count}, Pos={pos_count}")

    for p, y_hat, y in zip(probs_tensor, preds, gts_tensor):
        _update_metrics(metrics, p, y_hat, y, split)
        cm.update(y_hat, y)

    plot_confmat(
        cm.compute().cpu().numpy(),
        f"{split.title()} set - Binary Confusion Matrix",
        save_path / f"{split}_confusion_matrix.png",
    )

    plot_tsne(
        audio_emb,
        av_labels,
        f"{split.title()} set - Audio Projection",
        save_path / f"{split}_audio_proj.png",
    )
    plot_tsne(
        video_emb,
        av_labels,
        f"{split.title()} set - Video Projection",
        save_path / f"{split}_video_proj.png",
    )
    plot_tsne(
        fused_emb,
        av_labels,
        f"{split.title()} set - Fused Embedding",
        save_path / f"{split}_fused.png",
    )
    
    mean_inf_time_ms = (total_inference_time / max(n_samples, 1)) * 1000
    print(f"[{split.upper()}] Mean Inference Time: {mean_inf_time_ms:.2f} ms/sample")

    # Save to object for metrics output later
    if split == "val":
        _eval_split.val_inf_time_ms = mean_inf_time_ms
    elif split == "test":
        _eval_split.test_inf_time_ms = mean_inf_time_ms


def save_metrics(metrics, path):
    for split, split_metrics in metrics.items():
        for k, v in split_metrics.items():
            v = v.compute()
            v = v.item()
            metrics[split][k] = v

    if hasattr(_eval_split, 'val_eer'):
        if 'val' in metrics:
            metrics['val']['eer'] = _eval_split.val_eer
        else:
            metrics['val'] = {'eer': _eval_split.val_eer}

    if hasattr(_eval_split, 'val_best_f1'):
        metrics.setdefault('val', {})['best_f1'] = _eval_split.val_best_f1

    if hasattr(_eval_split, 'val_threshold'):
        metrics.setdefault('val', {})['threshold'] = _eval_split.val_threshold

    if hasattr(_eval_split, 'test_eer'):
        if 'test' in metrics:
            metrics['test']['eer'] = _eval_split.test_eer
        else:
            metrics['test'] = {'eer': _eval_split.test_eer}
            
    if hasattr(_eval_split, 'val_inf_time_ms'):
        metrics.setdefault("val", {})["mean_inference_time_ms"] = round(_eval_split.val_inf_time_ms, 2)

    if hasattr(_eval_split, 'test_inf_time_ms'):
        metrics.setdefault("test", {})["mean_inference_time_ms"] = round(_eval_split.test_inf_time_ms, 2)


    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def evaluate(dm, pipe, metrics, save_path, task, max_val_samples=None, max_test_samples=None):

    val_loader = dm.val_dataloader()
    _eval_split(val_loader, pipe, metrics, "val", task, save_path, max_samples=max_val_samples)
    print("Validation Set Metrics")
    for k, v in metrics["val"].items():
        print(f"{k}: {v.compute().item(): .3f}")
    if "mean_inference_time_ms" in metrics["val"]:
        print(f"Mean Inference Time (VAL): {metrics['val']['mean_inference_time_ms']:.2f} ms/sample")

    print("\n" + "-" * 50 + "\n")

    test_loader = dm.test_dataloader()
    _eval_split(test_loader, pipe, metrics, "test", task, save_path, max_samples=max_test_samples)
    print("Test Set Metrics")
    for k, v in metrics["test"].items():
        print(f"{k}: {v.compute().item(): .3f}")
    if "mean_inference_time_ms" in metrics["test"]:
        print(f"Mean Inference Time (TEST): {metrics['test']['mean_inference_time_ms']:.2f} ms/sample")


if __name__ == "__main__":
    args = parse_args()
    dm, pipe, path = setup(args)

    save_path = path / "results" / args.dataset
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    metrics = get_metrics(args.task)

    evaluate(
        dm,
        pipe,
        metrics,
        save_path,
        args.task,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    save_metrics(metrics, save_path / "metrics.json")