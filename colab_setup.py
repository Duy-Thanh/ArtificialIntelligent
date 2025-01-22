import os
import subprocess
from pathlib import Path
from datasets import load_dataset

def setup_colab():
    """Set up Google Colab environment for training"""
    print("Setting up Colab environment...")
    
    # Install required packages
    subprocess.run(["pip", "install", "-q", "transformers", "nltk", "pandas", "pyarrow", "tqdm", "datasets"])
    
    # Download dataset from Hugging Face
    dataset = load_dataset("your_username/your_dataset")
    dataset.save_to_disk("datasets")
    
    # Clone the repository if not already present
    if not Path("ArtificialIntelligent").exists():
        subprocess.run(["git", "clone", "https://github.com/Duy-Thanh/ArtificialIntelligent.git"])
        os.chdir("ArtificialIntelligent")
    
    # Create necessary directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("datasets").mkdir(exist_ok=True)
    
    # Download dataset files
    dataset_urls = [
        "https://your-storage/train-00000-of-00002.parquet",
        "https://your-storage/train-00001-of-00002.parquet",
        "https://your-storage/validation-00000-of-00001.parquet"
    ]
    
    for url in dataset_urls:
        filename = url.split('/')[-1]
        output_path = f"datasets/{filename}"
        if not Path(output_path).exists():
            print(f"Downloading {filename}...")
            subprocess.run(["wget", "-O", output_path, url])
    
    print("Setup complete!")

if __name__ == "__main__":
    setup_colab() 