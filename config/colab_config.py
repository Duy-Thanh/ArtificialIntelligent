"""Configuration for Google Colab environment"""

COLAB_CONFIG = {
    "batch_size": 12,          # Keep batch size
    "gradient_accumulation_steps": 8,  # Keep accumulation steps
    "num_workers": 2,          # Reduced to avoid overhead
    "prefetch_factor": 2,      # Reduced to avoid memory pressure
    "training_config": {
        "learning_rate": 6e-4,
        "warmup_steps": 250,
        "weight_decay": 0.01,
        "max_grad_norm": 0.5
    },
    "model_config": {
        "n_layer": 6,
        "n_head": 12,
        "n_embd": 384,
        "dropout": 0.15,
        "layer_norm_epsilon": 1e-5
    },
    "data_config": {
        "preprocessing_chunk_size": 10000,  # Added for parallel processing
        "max_samples": None,  # Remove sample limit
        "validate_data": False,  # Skip validation for speed
        "memory_efficient": True
    }
} 