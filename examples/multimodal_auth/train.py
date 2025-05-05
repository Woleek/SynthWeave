import json
import os
import re
import torch
import argparse
from typing import Dict, get_args, Literal
from unimodal import AdaFace, ReDimNet
from pipe import MultiModalAuthPipeline, ImagePreprocessor, AudioPreprocessor
from synthweave.utils.datasets import get_datamodule
from synthweave.utils.fusion import get_fusion
from synthweave.fusion import FusionType
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import (
    Accuracy,
    AveragePrecision,
    Precision,
    Recall,
    F1Score,
    ConfusionMatrix,
)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional.classification import binary_roc


def _far_frr(cm: torch.Tensor):
    tn, fp, fn, tp = cm.flatten()
    far = fp / (fp + tn + 1e-8)  # False-Accept
    frr = fn / (fn + tp + 1e-8)  # False-Reject
    return far, frr


def _eer(probs: torch.Tensor, labels: torch.Tensor):
    fpr, tpr, _ = binary_roc(probs, labels)
    frr = 1 - tpr
    idx = torch.argmin(torch.abs(fpr - frr))
    return 0.5 * (fpr[idx] + frr[idx])


class DeepfakeDetector(pl.LightningModule):
    def __init__(
        self,
        pipeline,
        detection_task: Literal["binary", "fine-grained"] = "binary",
        lr: float = 1e-4,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.task = detection_task
        self.lr = lr

        self.metrics: Dict[str, Dict[str, torch.nn.Module]] = {}
        metric_kwargs = (
            {"task": "binary"}
            if self.task == "binary"
            else {"task": "multiclass", "num_classes": 4, "average": "macro"}
        )

        base = dict(
            acc=Accuracy(**metric_kwargs),
            ap=AveragePrecision(**metric_kwargs),
            prec=Precision(**metric_kwargs),
            rec=Recall(**metric_kwargs),
            f1=F1Score(**metric_kwargs),
        )

        for split in ("train", "val", "test"):
            self.metrics[split] = {k: v.clone() for k, v in base.items()}

        if self.task == "binary":
            self.confmat_val = ConfusionMatrix(task="binary", num_classes=2)
            self.confmat_test = ConfusionMatrix(task="binary", num_classes=2)
            # For EER
            self._probs_val, self._labels_val = [], []
            self._probs_test, self._labels_test = [], []

    def setup(self, stage: str):
        # Ensure metrics are on correct device after model is moved
        device = self.device
        for split_metrics in self.metrics.values():
            for metric in split_metrics.values():
                metric.to(device)
        if self.task == "binary":
            self.confmat_val = self.confmat_val.to(device)
            self.confmat_test = self.confmat_test.to(device)

    def forward(self, batch):
        return self.pipeline(batch)

    def _get_labels(self, batch):
        return (
            batch["metadata"]["label"]
            if self.task == "binary"
            else batch["metadata"]["av"]
        )

    def _update_metrics(self, stage, preds, prob, labels):
        m = self.metrics[stage]

        m["acc"].update(preds, labels)
        m["ap"].update(prob, labels)
        m["prec"].update(preds, labels)
        m["rec"].update(preds, labels)
        m["f1"].update(preds, labels)

        if self.task == "binary":
            if stage == "val":
                self.confmat_val.update(preds, labels)
                self._probs_val.append(prob.detach())
                self._labels_val.append(labels.detach())
            elif stage == "test":
                self.confmat_test.update(preds, labels)
                self._probs_test.append(prob.detach())
                self._labels_test.append(labels.detach())

    def _log_epoch_metrics(self, stage):
        m = self.metrics[stage]
        self.log_dict(
            {f"{stage}/{k}": v.compute() for k, v in m.items()}, prog_bar=False
        )
        for v in m.values():
            v.reset()

        if self.task == "binary" and stage in ("val", "test"):
            cm = (
                self.confmat_val.compute()
                if stage == "val"
                else self.confmat_test.compute()
            )
            probs = (
                torch.cat(self._probs_val)
                if stage == "val"
                else torch.cat(self._probs_test)
            )
            labels = (
                torch.cat(self._labels_val)
                if stage == "val"
                else torch.cat(self._labels_test)
            )

            if probs.numel() > 0 and labels.numel() > 0:
                far, frr = _far_frr(cm)
                eer = _eer(probs, labels)

                self.log_dict(
                    {f"{stage}/far": far, f"{stage}/frr": frr, f"{stage}/eer": eer},
                    prog_bar=False,
                )

            # Reset after logging
            if stage == "val":
                self.confmat_val.reset()
                self._probs_val.clear()
                self._labels_val.clear()
            else:
                self.confmat_test.reset()
                self._probs_test.clear()
                self._labels_test.clear()

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def _step(self, batch, stage):
        y = self._get_labels(batch)
        logits = self(batch)["logits"]

        if self.task == "binary":
            prob = logits.squeeze(-1)
            loss = F.binary_cross_entropy(prob, y.float())
            preds = prob > 0.5
        else:
            prob = logits
            loss = F.cross_entropy(prob, y)
            preds = torch.argmax(prob, dim=1)

        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"))
        self._update_metrics(stage, preds, prob, y)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def parse_args():

    def clip_to_type(value):
        if value == "min":
            return "min"
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"clip_to must be an integer or 'min', got: {value}"
            )

    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--task", type=str, default="binary", choices=["binary", "fine-grained"]
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from checkpoint",
    )

    # Dataset
    parser.add_argument("--window_len", type=int, default=4)
    parser.add_argument("--window_step", type=int, default=1)
    parser.add_argument("--clip_mode", type=str, default="id", choices=["id", "idx"])
    parser.add_argument(
        "--clip_to",
        type=clip_to_type,
        default="min",
        help="clip_to must be an integer or 'min'",
    )
    parser.add_argument(
        "--clip_selector", type=str, default="random", choices=["first", "random"]
    )
    parser.add_argument(
        "--preprocessed", action="store_true", help="Use preprocessed dataset"
    )
    parser.add_argument("--encoded", action="store_true", help="Use encoded dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/DeepSpeak_v1_1/",
        help="Path to the dataset directory",
    )

    # Fusion
    parser.add_argument(
        "--fusion", type=str, default="CFF", choices=get_args(FusionType)
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=256,
        help="Output embedding dimension of the fusion module",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="Output projection dimension of the fusion module",
    )

    # Developement
    parser.add_argument("--dev", action="store_true", default=False)

    return parser.parse_args()


def main(args: argparse.Namespace):
    # PREPARE DATALODER
    if args.preprocessed or args.encoded:
        ds_kwargs = {
            "data_dir": args.data_dir,
            "preprocessed": True,
            "sample_mode": "single",
        }
    else:
        vid_proc = ImagePreprocessor(
            window_len=args.window_len,
            step=args.window_step,
            head_pose_dir="../../../models/head_pose",
        )  # TODO: handle no face detection error
        aud_proc = AudioPreprocessor(window_len=args.window_len, step=args.window_step)

        ds_kwargs = {
            "video_processor": vid_proc,
            "audio_processor": aud_proc,
            "mode": "minimal",
            "sample_mode": "single",
        }

    dm = get_datamodule(
        "DeepSpeak_v1_1",
        batch_size=args.batch_size,
        dataset_kwargs=ds_kwargs,
        sample_mode="single",  # single, sequence
        clip_mode=args.clip_mode,  # 'id', 'idx'
        clip_to=args.clip_to,  # 'min', int
        clip_selector=args.clip_selector,  # 'first', 'random'
    )

    # PREPARE PIPELINE
    if not args.encoded:
        aud_model = ReDimNet(freeze=True)
        img_model = AdaFace(path="../../../models/", freeze=True)

        models = {"audio": aud_model, "video": img_model}
    else:
        models = {"audio": torch.nn.Identity(), "video": torch.nn.Identity()}

    FUSION = args.fusion
    EMB_DIM = args.emb_dim

    fusion = get_fusion(
        fusion_name=FUSION,
        output_dim=EMB_DIM,
        modality_keys=["video", "audio"],
        out_proj_dim=args.proj_dim,
        # num_att_heads=4,
    )

    if args.task == "binary":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, 1), torch.nn.Sigmoid()
        )
    elif args.task == "fine-grained":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, 4), torch.nn.Softmax(dim=1)
        )

    pipe = MultiModalAuthPipeline(
        models=models,
        fusion=fusion,
        detection_head=detection_head,
        freeze_backbone=True,
    )

    print(
        f"""
    
    [PIPELINE]
        Fusion: {args.fusion}
        Projection Dim: {args.proj_dim}
        Embedding Dim: {args.emb_dim}
    
    [TRAINING]
        Detection Task: {args.task}
        Batch Size: {args.batch_size}
        Learning Rate: {args.lr:.2e}
    
    [DATASET]
        Preprocessed: {args.preprocessed or args.encoded}
        Encoded: {args.encoded}
        Source: {args.data_dir}
        Window Length: {args.window_len}
        Window Step: {args.window_step}
        Clip Mode: {args.clip_mode}
        Clip To: {args.clip_to}
        Clip Selector: {args.clip_selector}
    
    """
    )

    # PREPARE TRAINER
    run_name = f"{args.task}/{args.fusion}"
    log_base_dir = "logs"
    log_path = os.path.join(log_base_dir, run_name)

    if args.resume and not args.dev:
        # Get latest version_x directory
        versions = [
            d
            for d in os.listdir(log_path)
            if os.path.isdir(os.path.join(log_path, d)) and re.match(r"version_\d+", d)
        ]
        if not versions:
            raise RuntimeError(
                f"No version directories found in {log_path} to resume from."
            )

        latest_version = max(versions, key=lambda v: int(v.split("_")[1]))
        print(f"[INFO] Resuming from {latest_version}")

        logger = TensorBoardLogger(log_base_dir, name=run_name, version=latest_version)
        ckpt_path = "last"
    else:
        logger = TensorBoardLogger(log_base_dir, name=run_name)
        ckpt_path = None

    model = DeepfakeDetector(pipeline=pipe, detection_task=args.task, lr=args.lr)

    model_checkpoint = ModelCheckpoint(
        monitor="val/eer",
        mode="min",
        save_last=True,
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[model_checkpoint],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
        fast_dev_run=args.dev,
    )

    # Save args
    if not args.dev:
        log_dir = logger.log_dir
        logger.save()

        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    args = parse_args()
    main(args)
