import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
from datetime import datetime
import numpy as np
import torch
import math

logger = logging.getLogger(__name__)

class MetricsTracker:
    """Custom training metrics tracking system"""
    
    def __init__(self, save_dir: str = "metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': [],
            'grad_norm': [],
            'epoch_metrics': [],
            'memory_usage': []
        }
        
        # Create experiment timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.save_dir / f"experiment_{self.timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Set up logging
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging to file"""
        log_file = self.experiment_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    def update(self, metrics: Dict[str, Any]):
        """Update metrics during training"""
        for key, value in metrics.items():
            if key in self.metrics:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[key].append(value)
        
        # Calculate perplexity if loss is provided
        if 'train_loss' in metrics:
            perplexity = math.exp(metrics['train_loss'])
            self.metrics['perplexity'].append(perplexity)
        
        # Track GPU memory if available
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            self.metrics['memory_usage'].append(memory_used)
    
    def log_epoch(self, epoch_metrics: Dict[str, Any]) -> bool:
        """Log epoch-level metrics and check for early stopping"""
        self.metrics['epoch_metrics'].append(epoch_metrics)
        
        # Log to file
        logger.info(f"Epoch {epoch_metrics['epoch']}:")
        for key, value in epoch_metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Check for improvement
        current_val_loss = epoch_metrics['val_loss']
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.epochs_without_improvement = 0
            return True  # Model improved
        else:
            self.epochs_without_improvement += 1
            return False  # Model did not improve
    
    def should_stop_early(self, patience: int = 3) -> bool:
        """Check if training should stop early"""
        return self.epochs_without_improvement >= patience
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = self.experiment_dir / "metrics.json"
        
        # Calculate summary statistics
        summary = {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.metrics['train_loss'][-1],
            'mean_train_loss': np.mean(self.metrics['train_loss']),
            'std_train_loss': np.std(self.metrics['train_loss']),
            'total_epochs': len(self.metrics['epoch_metrics']),
            'peak_memory_mb': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        }
        
        # Add summary to metrics
        self.metrics['summary'] = summary
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self):
        """Generate and save training plots"""
        # Plot training and validation loss
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        
        # Plot perplexity
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['perplexity'])
        plt.xlabel('Steps')
        plt.ylabel('Perplexity')
        plt.title('Model Perplexity')
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['learning_rate'])
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        # Plot memory usage if available
        if self.metrics['memory_usage']:
            plt.subplot(2, 2, 4)
            plt.plot(self.metrics['memory_usage'])
            plt.xlabel('Steps')
            plt.ylabel('Memory (MB)')
            plt.title('GPU Memory Usage')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "training_metrics.png", dpi=300)
        plt.close()
        
        # Save loss curves in higher resolution
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics['train_loss'], label='Training Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Progress (Detailed)')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.experiment_dir / "loss_curves_detailed.png", dpi=300)
        plt.close() 