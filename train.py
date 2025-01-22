import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
from transformer_model import TransformerConfig, GPTModel
from dataset import load_dataset_split, get_dataset_info
from metrics import MetricsTracker
import psutil
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int = 3,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    save_dir: str = "checkpoints",
    start_epoch: int = 0
):
    """Train the model with validation and checkpointing"""
    
    # Initialize metrics tracker
    metrics = MetricsTracker()
    
    model = model.to(device)
    
    # Enable cuDNN benchmarking and TF32
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    best_val_loss = float('inf')
    
    # Initialize mixed precision training
    scaler = torch.amp.GradScaler('cuda')
    
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            total_train_loss = 0
            train_steps = 0
            
            train_pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
            
            # Pre-fetch data to GPU
            for step, batch in enumerate(train_pbar):
                # Move batch to device asynchronously
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(batch['input_ids'], batch['attention_mask'])
                    logits = outputs.view(-1, outputs.size(-1))
                    labels = batch['labels'].view(-1)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    loss = loss / gradient_accumulation_steps
                
                total_train_loss += loss.item() * gradient_accumulation_steps
                train_steps += 1
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Update progress bar less frequently
                if step % 50 == 0:
                    train_pbar.set_postfix({
                        'loss': loss.item() * gradient_accumulation_steps,
                        'lr': scheduler.get_last_lr()[0],
                        'gpu_mem': f"{torch.cuda.memory_allocated() / 1024**2:.0f}MB"
                    })
                
                # Log metrics
                metrics.update({
                    'train_loss': loss.item() * gradient_accumulation_steps,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            
            current_train_loss = total_train_loss / train_steps if train_steps > 0 else float('inf')
            
            # Validation
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    logits = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)
                    
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    total_val_loss += loss.item()
                    
                    val_pbar.set_postfix({'loss': total_val_loss / (step + 1)})
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            
            # Log epoch metrics
            metrics.log_epoch({
                'epoch': epoch + 1,
                'train_loss': current_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # Save checkpoint if best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, current_train_loss, avg_val_loss, save_dir / f'best_model.pt', is_best=True)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        current_train_loss = total_train_loss / train_steps if train_steps > 0 else float('inf')
        current_val_loss = float('inf')  # We don't have validation loss during interruption
        
        # Save checkpoint with current losses
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            current_train_loss, current_val_loss, 
            save_dir / 'interrupted_checkpoint.pt', 
            is_best=False
        )
        logger.info(f"Saved interrupt checkpoint with train loss: {current_train_loss:.4f}")
    
    # Save final metrics and plots
    metrics.save_metrics()
    metrics.plot_metrics()
    return metrics

def collate_fn(batch):
    """Custom collate function to ensure proper padding"""
    # All tensors should already be padded to max_length
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }

def load_checkpoint(save_dir: Path) -> tuple:
    """Load the best available checkpoint"""
    best_path = save_dir / "best_model.pt"
    interrupted_path = save_dir / "interrupted_checkpoint.pt"
    final_path = save_dir / "final_model.pt"
    
    def load_and_verify(path):
        checkpoint = torch.load(path)
        if 'config' in checkpoint:
            logger.info(f"Model config from checkpoint: {checkpoint['config']}")
        return checkpoint
    
    if best_path.exists():
        logger.info(f"Loading best checkpoint from {best_path}")
        return load_and_verify(best_path), "best"
    elif interrupted_path.exists():
        logger.info(f"Loading interrupted checkpoint from {interrupted_path}")
        return load_and_verify(interrupted_path), "interrupted"
    elif final_path.exists():
        logger.info(f"Loading final model from {final_path}")
        return {"model_state_dict": torch.load(final_path), "epoch": 0}, "final"
    
    return None, None

def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def main():
    # Initialize device and print GPU info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")
    else:
        logger.info("Using CPU - training will be slow")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load datasets with preprocessing and size limit for testing
    preprocessing_pipeline = [
        'basic_clean',
        'remove_urls',
        'remove_email',
        'remove_special',
        'normalize_whitespace'
    ]
    
    # Load appropriate config based on environment
    if is_colab():
        from config.colab_config import COLAB_CONFIG
        train_batch_size = COLAB_CONFIG["batch_size"]
        gradient_accumulation_steps = COLAB_CONFIG["gradient_accumulation_steps"]
        num_workers = COLAB_CONFIG["num_workers"]
        prefetch_factor = COLAB_CONFIG["prefetch_factor"]
        pin_memory = True
        persistent_workers = True
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=512,
            **COLAB_CONFIG["model_config"]
        )
    else:
        # Local config
        train_batch_size = 8
        gradient_accumulation_steps = 8
        num_workers = 4
        prefetch_factor = 4
        pin_memory = True
        persistent_workers = True
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            max_position_embeddings=512,
            n_layer=4,     
            n_head=8,
            n_embd=256,    
            dropout=0.1
        )
    
    # Initialize model with configuration
    model = GPTModel(config)
    
    # Load data with parallel processing first
    train_dataset = load_dataset_split(
        "train", 
        tokenizer,
        validate_data=True,
        memory_efficient=True,
        preprocessing_pipeline=preprocessing_pipeline,
        preprocessing_kwargs={'remove_special': {'keep_punctuation': True}},
        max_samples=1000,
        batch_size=32,
        num_workers=num_workers
    )
    
    val_dataset = load_dataset_split(
        "validation", 
        tokenizer,
        validate_data=True,
        memory_efficient=True,
        preprocessing_pipeline=preprocessing_pipeline,
        preprocessing_kwargs={'remove_special': {'keep_punctuation': True}},
        max_samples=100,
        batch_size=32
    )

    # Print dataset info
    logger.info("Training Dataset Info:", get_dataset_info(train_dataset))
    logger.info("Validation Dataset Info:", get_dataset_info(val_dataset))

    # Move model to GPU if available
    model = model.to(device)
    if torch.cuda.is_available():
        logger.info(f"Model moved to GPU. Memory used: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True,
        collate_fn=collate_fn,
        multiprocessing_context='spawn'
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=train_batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn
    )

    # Check for existing checkpoints
    save_dir = Path("checkpoints")
    checkpoint, checkpoint_type = load_checkpoint(save_dir)
    start_epoch = 0
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Create scheduler after dataloader is created
    total_steps = len(train_dataloader) * 5  # Now train_dataloader exists
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100,
        num_training_steps=total_steps
    )
    
    if checkpoint is not None:
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint_type == "best":
            logger.info(f"Resuming from epoch {start_epoch} with validation loss: {checkpoint['val_loss']:.4f}")
        elif checkpoint_type == "interrupted":
            logger.info(f"Resuming from interrupted epoch {start_epoch}")
        else:
            logger.info("Loaded final model state, starting from epoch 0")

    # Optimize for RTX 3060 8GB with parallel processing
    if torch.cuda.is_available():
        train_batch_size = 8        # Keep small batch size to prevent OOM
        gradient_accumulation_steps = 8
        num_workers = 4             # Increased workers for parallel processing
        prefetch_factor = 4         # Increased prefetch factor
        pin_memory = True
        persistent_workers = True
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Train with GPU-optimized parameters
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=5,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
        save_dir="checkpoints",
        start_epoch=start_epoch
    )

    # Save final model
    torch.save(model.state_dict(), 'checkpoints/final_model.pt')
    logger.info("Training completed!")

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, save_path, is_best=False):
    """Save a checkpoint with all necessary info for resuming training"""
    checkpoint = {
        'epoch': epoch + 1,  # Save next epoch to start from
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': model.config.__dict__,  # Save model configuration
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved {'best' if is_best else 'checkpoint'} to {save_path}")

if __name__ == "__main__":
    main() 