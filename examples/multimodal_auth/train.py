from collections import defaultdict
import copy
import json
import os
import re
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import torch
import argparse
from typing import Any, Dict, get_args, Literal
from src.pipe import (
    MultiModalAuthPipeline,
    ImagePreprocessor,
    AudioPreprocessor,
    AdaFace,
    ReDimNet,
)
import torch.nn as nn
from sklearn.cluster import KMeans
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
    AUROC,
)
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.tuner import Tuner
from pytorch_lightning import seed_everything
import matplotlib
matplotlib.use("Agg")

from src.iil import (
    IdentityAttenuationLoss, GradientReversal,
    MIDiscriminator, mutual_info_loss, linear_probe_identity, tsne_and_ari
)

from src.adv_training import (
    CenterClusterLoss, WithinModalityCELoss, WithinModalityMarginLoss, JSDFeatureLoss, JSDPredictionLoss, cross_modal_contrastive_loss, cross_modality_regularization
)

from src.visualization import plot_tsne_figure

class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.4, num_classes: int = 1):
        super().__init__()
        self.classifier = nn.Sequential(
            # nn.Linear(input_dim, input_dim),
            # nn.LayerNorm(input_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(input_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )
        
        self.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class IdCoder:
    def __init__(self):
        self._to_idx = {}              # str  -> int
    def encode(self, ids):             # ids: list[str]
        idxs = []
        for s in ids:
            if s not in self._to_idx:          # new speaker
                self._to_idx[s] = len(self._to_idx)
            idxs.append(self._to_idx[s])
        return torch.tensor(idxs, dtype=torch.long)
   
   
class DeepfakeDetector(pl.LightningModule):
    def __init__(
        self,
        pipeline: MultiModalAuthPipeline,
        detection_task: Literal["binary", "fine-grained"] = "binary",
        lr: float = 1e-4,
        iil_mode: Literal["none", "crossdf", "friday", "whitening"] = "none",
        train_strategy: Literal["default", "FRADE", "MRDF_ce", "MRDF_marg", "JSD"] = "default",
    ):
        super().__init__()
        self.identity_buffer_size: int = 4000   # Max samples to accumulate
        self.max_tracked_ids = 20
        
        self.pipeline = pipeline
        self.task = detection_task
        self.lr = lr
        self.iil_mode = iil_mode
        self.train_strategy = train_strategy
        
        proj_dim = self.pipeline.fusion.proj_dim
        embed_dim = self.pipeline.fusion.output_dim
        
        if self.train_strategy == "FRADE":
            # Center cluster loss (FRADE)
            self.ccl = CenterClusterLoss(embed_dim=embed_dim, gamma2=0.02, top_p=0.8, num_centers=3)
            
        elif self.train_strategy == "MRDF_ce":
            # within‐modality regularization (MRDF_ce)
            self.wmr_aud = WithinModalityCELoss(proj_dim)
            self.wmr_vid = WithinModalityCELoss(proj_dim)
        elif self.train_strategy == "MRDF_marg":
            # within‐modality regularization (MRDF_marg)
            self.wmr_aud = WithinModalityMarginLoss()
            self.wmr_vid = WithinModalityMarginLoss()
            
        elif self.train_strategy == "JSD":
            self.jsd_pred = JSDPredictionLoss(dim=proj_dim)
            self.jsd_feat = JSDFeatureLoss()
            
            self.beta_pred = 0.5
            self.beta_feat = 0.5
        
        if self.iil_mode == "crossdf":
            self.lambda_dec = 1.0           # CrossDF paper
            self.beta_grl   = 1.0           # GRL strength
            
            self.grl       = GradientReversal(beta=self.beta_grl)
            self.mi_disc   = MIDiscriminator(dim=embed_dim)
            
        elif self.iil_mode == "friday":
            self.ia_loss   = IdentityAttenuationLoss()
            
            self.lambda_ia  = 10.0          # FRIDAY paper
            
            self.id_branch = copy.deepcopy(pipeline) 
            self.id_branch.head = nn.Linear(embed_dim, 156)  # 156 identities
                    
        # Person ID coders (for adversarial loss)
        self.id_coder = {split: IdCoder() for split in ("train", "val", "test")}

        # Caches for t-SNE, logging, etc.
        self._init_logging_buffers()
        
        # Identity leakage monitoring
        self._init_identity_buffers()

        # Metrics
        self.metrics = self._init_metrics()
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outs = self.pipeline(batch)
        return outs
    
    def _step(self, batch, stage, batch_idx):
        bs = batch["video"].size(0)
        
        out = self(batch)
        y = self._get_labels(batch)
        id_t = self.id_coder[stage].encode(batch["metadata"]["id_target"]).to(batch["video"].device)
        
        # ========== Losses ==========
        loss_terms = []
        
        # Deepfake binary loss
        if self.task == "binary":
            logits = out["logits"].squeeze(-1)  # shape (B,)
            cls_loss = F.binary_cross_entropy_with_logits(logits, y.float().clamp(0.05, 0.95))
        else:
            logits = out["logits"] # shape (B, 4)
            cls_loss = F.cross_entropy(logits, y.long(), label_smoothing=0.05)
        self.log(f"{stage}/cls_loss", cls_loss, prog_bar=False, batch_size=bs)
        loss_terms.append(cls_loss)
        
        # IIL mitigation
        if self.iil_mode == "crossdf":
            z_df, z_os = out["embedding"], out["id_embedding"]
            joint = self.mi_disc(self.grl(z_df), self.grl(z_os))
            perm  = torch.randperm(bs, device=self.device)
            neg   = self.mi_disc(self.grl(z_df[perm]), self.grl(z_os))
            dec   = self.lambda_dec * mutual_info_loss(joint, neg)
            self.log(f"{stage}/dec_loss", dec, batch_size=bs)
            loss_terms.append(dec)
            
        elif self.iil_mode == "friday":
            id_outs = self.id_branch(batch)
            ia = self.lambda_ia * self.ia_loss(out["embedding"], id_outs["embedding"].detach())
            self.log(f"{stage}/ia_loss", ia, batch_size=bs)
            loss_terms.append(ia)
            
            id_loss = F.cross_entropy(id_outs["logits"], id_t.long())
            self.log(f"{stage}/id_adv_loss", id_loss, prog_bar=False, batch_size=bs)
            loss_terms.append(id_loss * 0.5)
        
        # Advanced training strategies
        if self.train_strategy == "FRADE":
            # Cross‐modal contrastive loss (FRADE)
            cmc_loss = cross_modal_contrastive_loss(F.normalize(out["audio_proj"]), F.normalize(out["video_proj"]), y)
            self.log(f"{stage}/cmc_loss", cmc_loss, prog_bar=False, batch_size=bs)
            loss_terms.append(cmc_loss * 0.3)
            
            # Center‐Cluster Loss (FRADE)
            cc_loss = self.ccl(F.normalize(out["embedding"], p=2, dim=1), y)
            self.log(f"{stage}/cc_loss", cc_loss, prog_bar=False, batch_size=bs)
            loss_terms.append(cc_loss * 0.4)
            
        elif self.train_strategy in ("MRDF_ce", "MRDF_marg"):
            # cross‐modality regularization (MRDF)
            cmr_loss = cross_modality_regularization(
                out['audio_proj'], out['video_proj'], y
            )
            self.log(f"{stage}/cmr_loss", cmr_loss, prog_bar=False, batch_size=bs)
            loss_terms.append(cmr_loss)
            
            # within‐modality regularization (MRDF)
            aud_t = self._get_per_modality_binary_labels(batch, "audio")
            wmr_a_loss = self.wmr_aud(out['audio_proj'], aud_t)
            
            vid_t = self._get_per_modality_binary_labels(batch, "video")
            wmr_v_loss = self.wmr_vid(out['video_proj'], vid_t)
                
            wmr_loss = wmr_a_loss + wmr_v_loss
            self.log(f"{stage}/wmr_loss", wmr_loss, prog_bar=False, batch_size=bs)
            loss_terms.append(wmr_loss)
            
        elif self.train_strategy == "JSD":
            aud_t = self._get_per_modality_binary_labels(batch, "audio")
            vid_t = self._get_per_modality_binary_labels(batch, "video")
            
            # pred-space JSD
            jsd_pred = self.jsd_pred(out["audio_proj"], out["video_proj"], aud_t, vid_t)
            self.log(f"{stage}/jsd_pred", jsd_pred, batch_size=bs)
            loss_terms.append(self.beta_pred * jsd_pred)
            
            # feat-space JSD stays the same
            jsd_feat = self.jsd_feat(out["audio_proj"], out["video_proj"], aud_t, vid_t)
            self.log(f"{stage}/jsd_feat", jsd_feat, batch_size=bs)
            loss_terms.append(self.beta_feat * jsd_feat)


        # ========== Metrics ==========
        if self.task == "binary":
            prob = F.sigmoid(logits)
            preds = prob > 0.5
        else:
            prob = F.softmax(logits, dim=-1)
            preds = torch.argmax(prob, dim=-1)
        
        self._update_metrics(stage, preds.long(), prob, y)
        
        # id accuracy debug
        self._accumulate_identity_features(stage, out["embedding"], id_t)

        # ========== Embeddings for visualization ==========
        # # How many more rows can we still take?
        # remaining = 1000 - self.last_count[stage]
        # if remaining > 0:
        #     # Which samples in this batch have IDs ∈ self.tracked_identities[stage] ?
        #     #    tracked_ids is a Python list of ints, up to length=10.
        #     tracked_ids = self.tracked_identities[stage]
        #     if tracked_ids:
        #         # Build a boolean mask of shape (B,), True for rows to keep
        #         ids_tensor = id_t  # shape (B,), on same device as out["embedding"]
        #         try:
        #             mask = torch.isin(ids_tensor, torch.tensor(tracked_ids, device=ids_tensor.device))
        #         except AttributeError:
        #             mask = torch.zeros_like(ids_tensor, dtype=torch.bool)
        #             for uid in tracked_ids:
        #                 mask |= (ids_tensor == uid)

        #         # Extract indices where mask == True
        #         indices_all = torch.nonzero(mask, as_tuple=False).flatten()  # shape (#kept,)
        #         if indices_all.numel() > 0:
        #             if indices_all.numel() > remaining:
        #                 indices_to_take = indices_all[:remaining]
        #             else:
        #                 indices_to_take = indices_all

        #             emb_sel  = out["embedding"][indices_to_take].detach().cpu()   # (take, D)
        #             aud_sel  = out["audio_proj"][indices_to_take].detach().cpu()
        #             vid_sel  = out["video_proj"][indices_to_take].detach().cpu()
        #             pid_sel  = id_t[indices_to_take].detach().cpu()
        #             gt_sel   = y[indices_to_take].detach().cpu()
        #             av_sel   = batch["metadata"]["av"][indices_to_take].detach().cpu()
        #             pred_sel = preds[indices_to_take].detach().cpu()

        #             # Append to per-split lists
        #             self.last_emb[stage].append(  emb_sel  )
        #             self.last_aud[stage].append(  aud_sel  )
        #             self.last_vid[stage].append(  vid_sel  )
        #             self.last_pid[stage].append(  pid_sel  )
        #             self.last_gt[stage].append(   gt_sel   )
        #             self.last_av[stage].append(   av_sel   )
        #             self.last_preds[stage].append(pred_sel )

        #             # Update stored‐row counter
        #             taken = indices_to_take.numel()
        #             self.last_count[stage] += taken

        total_loss = torch.stack(loss_terms).sum()
        self.log(f"{stage}/loss", total_loss, prog_bar=True, batch_size=bs)
        return total_loss
    
    def on_train_start(self):
        
        # Only initialize once
        if hasattr(self, "_centers_initialized"):
            if self._centers_initialized:
                return

            # Grab one batch
            train_loader = self.trainer.datamodule.train_dataloader()
            batch = next(iter(train_loader))

            # Move all tensor‐fields of batch to the trainer’s device
            batch = self.transfer_batch_to_device(batch, self.device, 0)

            # Forward pass through pipeline to get embeddings
            outs = self.pipeline(batch)
            cls_emb = outs["embedding"]       # shape (B, D), on GPU
            labels  = self._get_labels(batch) # shape (B,), on GPU

            # Initialize centers via K-Means on those embeddings
            self.ccl.initialize_centers_from(cls_emb, labels, device=self.device)

            self._centers_initialized = True
        
    def _init_identity_buffers(self):
        splits = ("train", "val", "test")
        # For each split, track the first ten identity indices
        self.tracked_identities = {split: [] for split in splits}

        # For each split, a dict from id_idx -> list[torch.Tensor of embeddings]
        self.identity_features = {
            split: {} for split in splits
        }
        
    def _accumulate_identity_features(self, stage: str, embeddings: torch.Tensor, identity_ids: torch.Tensor):
        feats_cpu = embeddings.detach().cpu()
        ids_cpu   = identity_ids.detach().cpu()

        # find all unique IDs in this batch
        unique_ids_in_batch = torch.unique(ids_cpu).tolist()
        for uid in unique_ids_in_batch:
            if (
                uid not in self.tracked_identities[stage]
                and len(self.tracked_identities[stage]) < self.max_tracked_ids
            ):
                self.tracked_identities[stage].append(uid)
                self.identity_features[stage][uid] = []

        for i in range(feats_cpu.shape[0]):
            this_id = int(ids_cpu[i].item())
            if this_id in self.tracked_identities[stage]:
                self.identity_features[stage][this_id].append(feats_cpu[i])
       
    def _identity_probe_and_plot(self, stage: str):
        if self.last_count[stage] == 0:       # nothing stored
            return

        # gather and flatten
        emb  = torch.cat(self.last_emb[stage]).cpu().numpy()     # (N,d)
        pids = torch.cat(self.last_pid[stage]).cpu().numpy()     # (N,)

        # linear probe
        acc, _ = linear_probe_identity(emb, pids, max_iter=1000)
        self.log(f"{stage}/lin_probe_acc", acc)

        # t-SNE + ARI + figure
        ari, fig = tsne_and_ari(emb, pids)
        self.log(f"{stage}/tsne_ari", ari)

        # push figure to TensorBoard
        if fig is not None:
            self.logger.experiment.add_figure(
                f"{stage}/tsne_id", fig, global_step=self.current_epoch
            )
            plt.close(fig)
                 
    def _compute_identity_leakage_multibatch(self, stage: str) -> float:
        # Check at least one tracked ID and at least some embeddings
        id_dict = self.identity_features.get(stage, {})
        if not id_dict:
            return 0.0

        # Build two flat lists: features_list, labels_list
        features_list = []
        labels_list   = []
        for id_idx, tensor_list in id_dict.items():
            for t in tensor_list:
                features_list.append(t.unsqueeze(0))
                labels_list.append(torch.tensor([id_idx], dtype=torch.long))
        
        # If fewer than 3 distinct IDs OR fewer than 50 total samples, bail
        unique_ids = list(id_dict.keys())
        total_samples = sum(x.shape[0] for x in features_list)
        if len(unique_ids) < 3 or total_samples < 50:
            return 0.0

        # Concatenate into two tensors: all_features [N, D], all_labels [N]
        all_features = torch.cat(features_list, dim=0).cpu().numpy()  # shape (N, D)
        all_labels   = torch.cat(labels_list,   dim=0).cpu().numpy()  # shape (N,)

        # Decide on number of clusters
        n_clusters = min(len(unique_ids), 20)

        # Run KMeans on features
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(all_features)

            # Compute Adjusted Rand Index between true ID labels and cluster assignments
            ari_score = adjusted_rand_score(all_labels, cluster_labels)

            # Clamp to [0, 1]
            leakage_score = float(max(0.0, ari_score))
            return leakage_score

        except Exception as e:
            print(f"Error in compute_identity_leakage: {e}")
            return 0.0

    def _init_logging_buffers(self):
        self.last_emb   = defaultdict(list)
        self.last_aud   = defaultdict(list)
        self.last_vid   = defaultdict(list)
        self.last_pid   = defaultdict(list)
        self.last_gt    = defaultdict(list)
        self.last_av    = defaultdict(list)
        self.last_preds = defaultdict(list)

        self.last_count = {split: 0 for split in ("train", "val", "test")}

    def _init_metrics(self):
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
            auc=AUROC(**metric_kwargs),
        )
        return {split: {k: v.clone() for k, v in base.items()} for split in ("train", "val", "test")}

    def setup(self, stage: str):
        # Ensure metrics and buffers on correct device after model moves
        device = self.device
        for split_metrics in self.metrics.values():
            for metric in split_metrics.values():
                metric.to(device)

        # Create fine-grained label lookup table for AV
        label_encoders = self.trainer.datamodule.train_dataset.encoders
        self.av_encoder = label_encoders["av"]
        av_classes = label_encoders["av"].classes_
        lookup = torch.tensor(
            [[int(code[0]), int(code[1])] for code in av_classes],
            dtype=torch.uint8,
        )
        self.register_buffer("_av_lookup", lookup, persistent=False)

    def _get_labels(self, batch):
        return (
            batch["metadata"]["label"]
            if self.task == "binary"
            else batch["metadata"]["av"]
        )

    def _get_per_modality_binary_labels(self, batch, mod: Literal["video", "audio"]):
        col = 0 if mod == "audio" else 1
        av = batch["metadata"]["av"]
        return self._av_lookup[av, col]

    def _update_metrics(self, stage, preds, prob, labels):
        m = self.metrics[stage]
        m["acc"].update(preds, labels)
        m["ap"].update(prob, labels)
        m["prec"].update(preds, labels)
        m["rec"].update(preds, labels)
        m["f1"].update(preds, labels)
        m["auc"].update(prob, labels)

    def _log_epoch_metrics(self, stage):
        m = self.metrics[stage]
        self.log_dict(
            {f"{stage}/{k}": v.compute() for k, v in m.items()}, prog_bar=False
        )
        for v in m.values():
            v.reset()

    def on_train_epoch_start(self):
        self._init_logging_buffers()
        self._init_identity_buffers()
        
    def on_validation_epoch_start(self):
        # Reset validation identity buffers
        if 'val' in self.identity_features:
            self.identity_features['val'].clear()

    def _log_tsne(self, stage, type):
        if type == "audio":
            emb = torch.cat(self.last_aud[stage]).detach().cpu().numpy()
            labels = torch.cat(self.last_av[stage]).detach().cpu().numpy()
        elif type == "video":
            emb = torch.cat(self.last_vid[stage]).detach().cpu().numpy()
            labels = torch.cat(self.last_av[stage]).detach().cpu().numpy()
        else: # type == "fused"
            emb = torch.cat(self.last_emb[stage]).detach().cpu().numpy()
            labels = torch.cat(self.last_gt[stage]).detach().cpu().numpy()
        
        pids = torch.cat(self.last_pid[stage]).detach().cpu().numpy()
            
        preds = torch.cat(self.last_preds[stage]).detach().cpu().numpy()

        fig = plot_tsne_figure(emb, labels, preds, pids)

        # Lightning 2.x: log image to TensorBoard
        self.logger.experiment.add_figure(
            f"{stage}/tsne_{type}", fig, global_step=self.current_epoch
        )
        plt.close(fig)

    def on_train_epoch_end(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr)
        self._log_epoch_metrics("train")
        # if len(self.last_emb["train"]) > 0 and self.current_epoch % 5 == 0:
        #     self._log_tsne("train", "audio")
        #     self._log_tsne("train", "video")
        #     self._log_tsne("train", "fused")
            
        # Compute validation identity leakage
        train_identity_leakage = self._compute_identity_leakage_multibatch("train")
        self.log("train/identity_leakage_epoch", train_identity_leakage, prog_bar=False)

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")
        # if len(self.last_emb["val"]) > 0 and self.current_epoch % 5 == 0:
        #     self._log_tsne("val", "audio")
        #     self._log_tsne("val", "video")
        #     self._log_tsne("val", "fused")
        #     self._identity_probe_and_plot("val")
            
        # Compute validation identity leakage
        val_identity_leakage = self._compute_identity_leakage_multibatch("val")
        self.log("val/identity_leakage_epoch", val_identity_leakage, prog_bar=False)

    def on_test_epoch_end(self):
        self._log_epoch_metrics("test")
        # if len(self.last_emb["test"]) > 0 and self.current_epoch % 5 == 0:
        #     self._log_tsne("test", "audio")
        #     self._log_tsne("test", "video")
        #     self._log_tsne("test", "fused")
        #     self._identity_probe_and_plot("test")
            
        # Compute test identity leakage
        test_identity_leakage = self._compute_identity_leakage_multibatch("test")
        self.log("test/identity_leakage_epoch", test_identity_leakage, prog_bar=False)
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test", batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def on_after_backward(self):
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1e9)
        self.log("grad_norm/total", total_norm, prog_bar=False)


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
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the fusion module",
    )
    parser.add_argument(
        "--iil_mode",
        type=str,
        default="none",
        choices=["none", "crossdf", "friday", "whitening"],
    )
    parser.add_argument(
        "--train_strategy",
        type=str,
        default="default",
        choices=[
            "default",
            "FRADE",
            "MRDF_ce",
            "MRDF_marg",
            "JSD"
        ],
        help="Training strategy to use for the model.",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=["DeepSpeak_v1_1", "SWAN_DF"],
        required=True,
        help="Which dataset to load",
    )
    parser.add_argument("--window_len", type=int, default=4)
    parser.add_argument("--window_step", type=int, default=1)
    parser.add_argument("--clip_mode", type=str, default=None, choices=["id", "idx"])
    parser.add_argument(
        "--clip_to",
        type=clip_to_type,
        default=None,
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
        required=True,
        help="Path to the dataset directory",
    )

    # Fusion
    parser.add_argument(
        "--fusion", type=str, required=True, choices=get_args(FusionType)
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
        default=512,
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
        balance_classes=(args.dataset == "SWAN_DF"),
        num_workers=os.cpu_count()//2,
        encode_ids=False,
        drop_last_batch=True,
    )

    # PREPARE PIPELINE
    if not args.encoded:
        aud_model = ReDimNet(freeze=True)
        img_model = AdaFace(path="../../../models/", freeze=True, model_type='ir50')

        models = {"audio": aud_model, "video": img_model}
    else:
        models = {"audio": torch.nn.Identity(), "video": torch.nn.Identity()}

    FUSION = args.fusion
    EMB_DIM = args.emb_dim

    fusion = get_fusion(
        fusion_name=FUSION,
        output_dim=EMB_DIM,
        modality_keys=["video", "audio"],
        input_dims={
            "video": 512,  # AdaFace output dim
            "audio": 192,  # ReDimNet output dim
        },
        out_proj_dim=args.proj_dim,
        num_att_heads=2,  # only for attention-based fusions
        n_layers=2,  # only for MMD
        dropout=args.dropout,
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
        iil_mode = args.iil_mode,
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
        iil_mode = args.iil_mode,
        train_strategy = args.train_strategy,
    )
    
    print(model)

    model_checkpoint = ModelCheckpoint(
        monitor="val/auc",
        mode="max",
        save_last=True,
        save_top_k=1,
    )
            
    early = EarlyStopping(monitor="val/auc", mode="max", patience=5, min_delta=1e-3)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[model_checkpoint, early],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
        # gradient_clip_val=2.0,
        fast_dev_run=args.dev,
    )

    # Run LR finder
    # model.eval()
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, datamodule=dm, update_attr='lr', early_stop_threshold=None)
    # suggested_lr = lr_finder.suggestion()
    # print(f"[INFO] Setting model learning rate: {suggested_lr:.2e}")
    # args.lr = suggested_lr
    # model.train()

    # Save args
    if not args.dev:
        log_dir = logger.log_dir
        logger.save()

        with open(os.path.join(log_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

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
        Dropout: {args.dropout}
        IIL Mode: {args.iil_mode}
        Training Strategy: {args.train_strategy}
    
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

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)

    if not args.dev:
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    seed_everything(seed=42)

    args = parse_args()
    main(args)
