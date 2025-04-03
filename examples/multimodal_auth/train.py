import torch
import argparse
from typing import get_args, Literal
from unimodal import AdaFace, ReDimNet
from pipe import MultiModalAuthPipeline, ImagePreprocessor, AudioPreprocessor
from synthweave.utils.datasets import get_datamodule
from synthweave.utils.fusion import get_fusion
from synthweave.fusion import FusionType
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy, AUROC
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

class DeepfakeDetector(pl.LightningModule):
    def __init__(
        self, 
        pipeline,
        
        detection_task=Literal["binary", "fine-grained"],
        lr=1e-4
    ):
        super().__init__()
        self.pipeline = pipeline
        self.task = detection_task
        self.lr = lr
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(task="binary")
        
        self.example_input_array= ({
            "video": torch.randn([4, 3, 112, 112]),
            "audio": torch.randn([4, 1, 64000]),
            "metadata": [{"label": torch.randn([4]), "av": torch.randn([4])}]
        },)

    def forward(self, batch):
        return self.pipeline(batch)
    
    def _get_labels(self, batch):
        if self.task == "binary":
            return batch["metadata"]['label'] # 0, 1
        elif self.task == "fine-grained":
            return batch["metadata"]['av'] # 0, 1, 2, 3

    def training_step(self, batch: dict, batch_idx):
        y = self._get_labels(batch)
        if self.task == "binary":
            logits = self(batch)["logits"].squeeze()
            loss = F.binary_cross_entropy(logits, y.float())
            preds = logits > 0.5
            
        elif self.task == "fine-grained":
            logits = self(batch)["logits"]
            loss = F.cross_entropy(logits, y.long())
            preds = torch.argmax(logits, dim=1)
            
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.accuracy(preds, y.int()))
        self.log("train_auroc", self.auroc(logits, y.int()))
        
        return loss

    def validation_step(self, batch: dict, batch_idx):
        y = self._get_labels(batch)
        if self.task == "binary":
            logits = self(batch)["logits"].squeeze()
            loss = F.binary_cross_entropy(logits, y.float())
            preds = logits > 0.5
            
        elif self.task == "fine-grained":
            logits = self(batch)["logits"]
            loss = F.cross_entropy(logits, y.long())
            preds = torch.argmax(logits, dim=1)
            
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(preds, y.int()))
        self.log("val_auroc", self.auroc(logits, y.int()), prog_bar=True)
        
    def test_step(self, batch: dict, batch_idx):
        y = self._get_labels(batch)
        if self.task == "binary":
            logits = self(batch)["logits"].squeeze()
            loss = F.binary_cross_entropy(logits, y.float())
            preds = logits > 0.5
            
        elif self.task == "fine-grained":
            logits = self(batch)["logits"]
            loss = F.cross_entropy(logits, y.long())
            preds = torch.argmax(logits, dim=1)
            
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy(preds, y.int()))
        self.log("test_auroc", self.auroc(logits, y.int()))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "fine-grained"])
    
    # Fusion
    parser.add_argument("--fusion", type=str, default="CFF", choices=get_args(FusionType))
    
    # Developement
    parser.add_argument("--dev", action="store_true", default=False)
    
    return parser.parse_args()
    
def main(args: argparse.Namespace):
    # PREPARE DATALODER
    vid_proc = ImagePreprocessor(window_len=4, step=1) #TODO: handle no face detection error
    aud_proc = AudioPreprocessor(window_len=4, step=1)

    ds_kwargs = {
        'video_processor': vid_proc, 'audio_processor': aud_proc, 'mode': 'minimal'
    }

    dm = get_datamodule("DeepSpeak_v1", batch_size=args.batch_size, dataset_kwargs=ds_kwargs, 
                        sample_mode='single', # single, sequence
                        clip_mode='id', # 'id', 'idx'
                        clip_to='min', # 'min', int
                        clip_selector='random', # 'first', 'random'
    )
    
    # PREPARE PIPELINE
    aud_model = ReDimNet(
        freeze=True
    )

    img_model = AdaFace(
        path='../../../models/',
        freeze=True
    )
    
    FUSION = args.fusion
    EMB_DIM = 256

    fusion = get_fusion(
        fusion_name=FUSION,
        output_dim=EMB_DIM,
        modality_keys=["video", "audio"],
        out_proj_dim=256,
        # num_att_heads=4,
    )
    
    if args.task == "binary":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, 1),
            torch.nn.Sigmoid()
        )
    elif args.task == "fine-grained":
        detection_head = torch.nn.Sequential(
            torch.nn.Linear(EMB_DIM, 4),
            torch.nn.Softmax(dim=1)
        )
    
    pipe = MultiModalAuthPipeline(
        models={
            'audio': aud_model,
            'video': img_model
        },
        fusion=fusion,
        detection_head=detection_head,
        freeze_backbone=True
    )
    
    # PREPARE TRAINER
    run_name = f"{args.task}/{args.fusion}"
    logger = TensorBoardLogger("logs", name=run_name)

    model = DeepfakeDetector(
        pipeline=pipe, 
        detection_task=args.task,
        lr=args.lr
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[ModelCheckpoint(monitor="val_auroc", mode="max")],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        
        fast_dev_run=args.dev,
    )
    
    print(
    f"""
    
    [PIPELINE]
    Fusion: {args.fusion}
    
    [TRAINING]
    Detection Task: {args.task}
    Batch Size: {args.batch_size}
    Learning Rate: {args.lr:.2e}
    
    [DATASET]
    
    """
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    
    args = parse_args()
    main(args)
