"""Configuration for Google Colab environment"""

COLAB_CONFIG = {
    "batch_size": 32,  # Colab usually has more GPU memory
    "gradient_accumulation_steps": 2,
    "num_workers": 2,
    "prefetch_factor": 4,
    "model_config": {
        "n_layer": 6,
        "n_head": 8,
        "n_embd": 512,
    }
} 