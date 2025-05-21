import json
import os
import re
import torch
import argparse
from typing import Dict, get_args, Literal
from src.pipe import MultiModalAuthPipeline, ImagePreprocessor, AudioPreprocessor, AdaFace, ReDimNet
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

from pytorch_metric_learning.losses import SupConLoss, CrossBatchMemory, ArcFaceLoss
from pytorch_lightning import seed_everything


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
        pipeline: MultiModalAuthPipeline,
        detection_task: Literal["binary", "fine-grained"] = "binary",
        lr: float = 1e-4,
        cls_loss_weight: float = 1.0,
        contr_loss_fused_weight: float = 0.0,
        contr_loss_mod_weight: float = 0.0,
        aam_loss_weight: float = 0.0,
        lr_scheduler_name: str = None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.add_module("pipeline", pipeline)
        self.task = detection_task
        self.lr = lr
        self.lr_scheduler_name = lr_scheduler_name
        
        self.loss_w = {
            "cls": cls_loss_weight,
            "contr_fused": contr_loss_fused_weight,
            "contr_mod": contr_loss_mod_weight,
            "aam": aam_loss_weight,
        }

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

    def on_after_backward(self):
        if self.global_step % 100 == 0:  # every 100 steps
            # for n, p in self.named_parameters():
            #     if p.grad is not None:
            #         self.logger.experiment.add_scalar(
            #             f"grad_norm/{n}", p.grad.norm(), self.global_step
            #         )

            # 1) log grad-to-weight ratio  (helps detect layers that barely move)
            for n, p in self.named_parameters():
                if p.grad is None:
                    continue
                ratio = (p.grad.norm() / (p.data.norm() + 1e-12)).item()
                self.logger.experiment.add_scalar(f"g2w/{n}", ratio, self.global_step)

            # 2) histogram of gradients for one batch (nan / inf check)
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if p.grad is not None:
                        self.logger.experiment.add_histogram(
                            f"grad_dist/{n}", p.grad, self.global_step
                        )

    def setup(self, stage: str):
        # Ensure metrics are on correct device after model is moved
        device = self.device
        for split_metrics in self.metrics.values():
            for metric in split_metrics.values():
                metric.to(device)
        if self.task == "binary":
            self.confmat_val = self.confmat_val.to(device)
            self.confmat_test = self.confmat_test.to(device)

        # Losses
        for loss in self.loss_w:
            if self.loss_w[loss] > 0.0:
                if loss == "cls":
                    if self.task == "binary":
                        self.cls_loss = torch.nn.BCELoss()
                    else:
                        self.cls_loss = torch.nn.CrossEntropyLoss()
                elif loss == "contr_fused":
                    self.contr_loss_fused = SupConLoss()
                    # supcon_fused = SupConLoss()
                    # self.contr_loss_fused = CrossBatchMemory(
                    #     loss=supcon_fused,
                    #     embedding_size=self.pipeline.fusion.output_dim,
                    #     memory_size=512 # queue size
                    # )
                elif loss == "contr_mod":
                    self.contr_loss_mod = SupConLoss()
                    # supcon_mod = SupConLoss()
                    # self.contr_loss_mod = CrossBatchMemory(
                    #     loss=supcon_mod,
                    #     embedding_size=self.pipeline.fusion.proj_dim,
                    #     memory_size=512 # queue size
                    # )
                elif loss == "aam":
                    self.aam_loss = ArcFaceLoss(
                        num_classes=2 if self.task == "binary" else 4,
                        embedding_size=self.pipeline.fusion.output_dim,
                        margin=0.5,
                        scale=30,
                    )

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

    # def on_train_epoch_start(self):
    # Reset contrastive loss queues
    # if self.loss_w["contr_fused"] > 0.0:
    #     self.contr_loss_fused.reset_queue()
    # if self.loss_w["contr_mod"] > 0.0:
    #     self.contr_loss_mod.reset_queue()

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr)
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")

    def _step(self, batch, stage, batch_idx):
        y = self._get_labels(batch)
        out = self(batch)

        # SupCon loss
        fused_contr_loss = 0.0
        if self.loss_w["contr_fused"] > 0.0:
            emb = out["embedding"]
            emb_norm = F.normalize(emb, dim=-1)
            fused_contr_loss = self.contr_loss_fused(emb_norm, y)

        mod_contr_loss = 0.0
        if self.loss_w["contr_mod"] > 0.0:
            for mod in self.pipeline.fusion.modalities:
                mod_emb = out[f"{mod}_proj"]
                mod_emb_norm = F.normalize(mod_emb, dim=-1)
                mod_contr_loss += self.contr_loss_mod(mod_emb_norm, y)
        mod_contr_loss /= len(self.pipeline.fusion.modalities)

        # BCE / CE loss
        logits = out["logits"]

        if self.task == "binary":
            prob = logits.squeeze(-1)
            # cls_loss = F.binary_cross_entropy(prob, y.float())
            cls_loss = self.cls_loss(prob, y.float())
            preds = prob > 0.5
        else:
            prob = logits
            # cls_loss = F.cross_entropy(prob, y)
            cls_loss = self.cls_loss(prob, y)
            preds = torch.argmax(prob, dim=1)
            
        # AAM loss
        aam_loss = 0.0
        if self.loss_w["aam"] > 0.0:
            aam_loss = self.aam_loss(
                out["embedding"], y.long()
            )

        self._update_metrics(stage, preds, prob, y)

        # Total loss
        loss = (
            cls_loss * self.loss_w["cls"]
            + fused_contr_loss * self.loss_w["contr_fused"]
            + mod_contr_loss * self.loss_w["contr_mod"]
            + aam_loss * self.loss_w["aam"]
        )

        if stage == "train":
            # Log individual losses
            (
                self.log(f"{stage}/cls_loss", cls_loss)
                if self.loss_w["cls"] > 0.0
                else None
            )
            (
                self.log(f"{stage}/contr_fused_loss", fused_contr_loss)
                if self.loss_w["contr_fused"] > 0.0
                else None
            )
            (
                self.log(f"{stage}/contr_mod_loss", mod_contr_loss)
                if self.loss_w["contr_mod"] > 0.0
                else None
            )
            (
                self.log(f"{stage}/aam_loss", aam_loss)
                if self.loss_w["aam"] > 0.0
                else None
            )

        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test", batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

        if self.lr_scheduler_name is None:
            return optimizer

        elif self.lr_scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr * 0.1,
            )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        elif self.lr_scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=5,  # TODO: test other values
                threshold=0.001
            )

            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/f1",
                "strict": True,
            }
        else:
            raise ValueError(f"Unknown scheduler name: {self.scheduler_name}")

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--task", type=str, required=True, choices=["binary", "fine-grained"]
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--cls_loss_w",
        type=float,
        default=1.0,
        help="Weight for classification loss",
    )
    parser.add_argument(
        "--contr_loss_mod_w",
        type=float,
        default=0.05,
        help="Weight for contrastive loss on individual modalities",
    )
    parser.add_argument(
        "--contr_loss_fused_w",
        type=float,
        default=0.1,
        help="Weight for contrastive loss on fused features",
    )
    parser.add_argument(
        "--aam_loss_w",
        type=float,
        default=0.2,
        help="Weight for AAM loss",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the fusion module",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["cosine", "plateau", None],
        help="Learning rate scheduler",
    )

    # Dataset
    parser.add_argument("--dataset", choices=["DeepSpeak_v1_1", "SWAN_DF"],
                   required=True, help="Which dataset to load")
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
        "--fusion", type=str, required=True, choices=get_args(FusionType)
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=512,
        help="Output embedding dimension of the fusion module",
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=1024,
        help="Output projection dimension of the fusion module",
    )

    # Developement
    parser.add_argument("--dev", action="store_true", default=False)

    return parser.parse_args()


def main(args: argparse.Namespace):
    # PREPARE DATALODER
    if args.encoded:
        args.preprocessed = True
    
    if args.preprocessed or args.encoded:
        ds_kwargs = {
            "data_dir": args.data_dir,
            "preprocessed": True,
            "sample_mode": "single",
        }
        
        if args.dataset == "SWAN_DF":
            ds_kwargs["av_codes"] = ["00", "11"]
    else:
        vid_proc = ImagePreprocessor(
            window_len=args.window_len,
            step=args.window_step,
            head_pose_dir="../../../models/head_pose",
        )
        aud_proc = AudioPreprocessor(window_len=args.window_len, step=args.window_step)

        ds_kwargs = {
            "video_processor": vid_proc,
            "audio_processor": aud_proc,
            "mode": "minimal",
            "sample_mode": "single",
        }

    dm = get_datamodule(
        args.dataset,
        batch_size=args.batch_size,
        dataset_kwargs=ds_kwargs,
        sample_mode="single",  # single, sequence
        clip_mode=args.clip_mode,  # 'id', 'idx'
        clip_to=args.clip_to,  # 'min', int
        clip_selector=args.clip_selector,  # 'first', 'random'
        balance_classes=(args.dataset == "SWAN_DF")
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
        num_att_heads=4,  # only for attention-based fusions
        n_layers=3,  # only for MMD
        dropout=args.dropout,
    )

    if args.task == "binary":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, 1), torch.nn.Sigmoid()
        )
    elif args.task == "fine-grained":
        assert args.dataset != "SWAN_DF", "Fine-grained task is not supported for SWAN_DF, as it has only RA-RV and FA-FV classes."
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
        Contrastive Loss: {True if (args.contr_loss_mod_w or args.contr_loss_fused_w) else False}
        Margin Loss: {True if args.aam_loss_w > 0.0 else False}
        Dropout: {args.dropout}
    
    [DATASET]
        Dataset: {args.dataset}
        Preprocessed: {args.preprocessed}
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
    log_base_dir = "logs"
    run_name = f"{args.dataset}/{args.task}/{args.fusion}"
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

    model = DeepfakeDetector(
        pipeline=pipe,
        detection_task=args.task,
        lr=args.lr,
        cls_loss_weight=args.cls_loss_w,
        contr_loss_fused_weight=args.contr_loss_fused_w,
        contr_loss_mod_weight=args.contr_loss_mod_w,
        aam_loss_weight=args.aam_loss_w,
        lr_scheduler_name=args.scheduler,
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val/auc",
        mode="max",
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

    if not args.dev:
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    seed_everything(seed=42)
    torch.set_float32_matmul_precision("medium")

    args = parse_args()
    main(args)
