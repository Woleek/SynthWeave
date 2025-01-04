import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import gc
import numpy as np
import random
import os

from tqdm import tqdm
try:
    # Check if the environment is a Jupyter notebook
    from IPython import get_ipython
    is_notebook = get_ipython() is not None and 'IPKernelApp' in get_ipython().config
except:
    is_notebook = False

# Use the appropriate tqdm version
if is_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer():
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = torch.device('cpu'),
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        progress_bar: bool = True,
        save_path: str = None,
        load_best_model_at_the_end: bool = True,
        early_stopping_patience: int = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)
        self.scheduler = scheduler
        self.progress_bar = progress_bar
        self.save_path = save_path
        self.load_best = load_best_model_at_the_end
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        
    def train(self, dataloader: DataLoader, epochs: int, eval_dataloader: DataLoader = None) -> float:    
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if self.progress_bar:
            iterator = tqdm(range(epochs), desc="Training Progress")
        else:
            iterator = range(epochs)
        
        for epoch in iterator:
            self.model.train()
            epoch_loss = 0.0
            
            if self.progress_bar:
                iterator.set_description(f"Training Progress [{epoch + 1}/{epochs}]")
            
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} 
                outputs = self.model(batch)
                if 'label' in batch:
                    loss = self.criterion(outputs, batch['label'])
                else:
                    loss = self.criterion(outputs)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
            if self.progress_bar:
                iterator.set_postfix(epoch=epoch + 1, lr=current_lr, loss=epoch_loss / len(dataloader))
                iterator.update(1)
                
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                
                if self.progress_bar:
                    iterator.set_postfix(epoch=epoch + 1, lr=current_lr, loss=epoch_loss / len(dataloader), val_loss=eval_loss)
                
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.early_stop_counter = 0
                    if self.save_path:
                        torch.save(self.model.state_dict(), self.save_path)
                else:
                    self.early_stop_counter += 1
                    
            if self.early_stopping_patience and self.early_stop_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered. Best validation loss: {self.best_loss:.4f}")
                break
                
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and eval_dataloader:
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
        if self.load_best and self.save_path:
            self.model.load_state_dict(torch.load(self.save_path, weights_only=True))
    
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} 
                outputs = self.model(batch)
                if 'label' in batch:
                    loss = self.criterion(outputs, batch['label'])
                else:
                    loss = self.criterion(outputs)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        predictions = []
        targets = []
        
        if self.progress_bar:
            dataloader = tqdm(dataloader, desc="Prediction Progress", leave=True)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} 
                outputs = self.model(batch)
                predictions.append(outputs)
                
                if 'label' in batch:
                    targets.append(batch['label'])
                
                if self.progress_bar:
                    dataloader.update(1)
        
        if targets:
            return torch.cat(predictions), np.concatenate(targets, axis=0)
        return torch.cat(predictions)

class BatchSizeFinder():
    def __init__(
        self,
        model: nn.Module,
        mode: str = "power",
        init_val: int = 2,
        max_trials: int = 25,
        varying_batches = False,
        device = torch.device('cpu')
        
    ) -> None:
        self.mode = mode
        self.init_val = init_val
        self.max_trials = max_trials
        self.model = model
        self.device = device
        self.varying_batches = varying_batches
        
    def find_batch_size(self, dataloader: DataLoader) -> int:
        self.dataloader = dataloader
        self.dataset_len = len(dataloader.dataset)
        if self.mode == "power":
            new_size = self._run_power_scaling()

        garbage_collection_cuda()

        print(f"Finished running finder with batch size: {new_size}")
        return new_size
            
    def _run_power_scaling(
        self
    ) -> int:
        """Batch scaling mode where the size is doubled at each iteration until an OOM error is encountered."""
        any_success = False
        new_value, _ = self._adjust_batch_size(factor=1.0, value=self.init_val)
        
        for _ in range(self.max_trials):
            garbage_collection_cuda()

            try:
                new_value, changed = self._adjust_batch_size(factor=2.0, value=new_value)
                
                if not changed:
                    break
                
                self._try_run()

                # Force the train dataloader to reset as the batch size has changed
                self._reinitialize_dataloader(new_value)
                any_success = True
            except RuntimeError as exception:
                if is_oom_error(exception):
                    # If we fail in power mode, half the size and return
                    garbage_collection_cuda()
                    new_value, _ = self._adjust_batch_size(factor=0.5, value=new_value)
                    # Force the train dataloader to reset as the batch size has changed
                    self._reinitialize_dataloader(new_value)
                    if any_success:
                        break
                else:
                    raise  # some other error not memory related

        return new_value
    
    def _adjust_batch_size(self, factor: float, value: int):
        changed = False
        new_value = int(value * factor)
        
        if new_value > self.dataset_len:
            return value, changed
        else:
            changed = True
            return new_value, changed
        
    def _reinitialize_dataloader(self, new_value: int) -> None:
        self.dataloader = DataLoader(self.dataloader.dataset, batch_size=new_value, collate_fn=self.dataloader.collate_fn)
    
    def _try_run(self):
        if self.varying_batches: # needed for varying batch sizes
            s = next(iter(self.dataloader))
            # find modality with biggest size
            modality = max(s, key=lambda x: s[x].flatten().shape[0] if isinstance(s[x], torch.Tensor) else 0)
            
            max_batch = 1
            for batch in self.dataloader:
                max_batch = max(max_batch, batch[modality].shape[0])

            # create sample tensor for each modality with size of max_batch
            big_batch = {modality: torch.rand((max_batch, *s[modality].shape[1:]), device=self.device) for modality in s if isinstance(s[modality], torch.Tensor)}
            
            self.model(big_batch)
            
        else:
            for batch in self.dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()} 
                self.model(batch)
                break

def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # Only handle OOM errors
            raise
        
def is_oom_error(exception: BaseException) -> bool:
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)

def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )

def is_cudnn_snafu(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )

def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )