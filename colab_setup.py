import os
import subprocess
from pathlib import Path

def setup_colab():
    """Set up Google Colab environment for training"""
    print("Setting up Colab environment...")
    
    # Install required packages
    subprocess.run(["pip", "install", "-q", "transformers", "nltk", "pandas", "pyarrow", "tqdm", "datasets"])
    
    # Clone the repository if not already present
    if not Path("ArtificialIntelligent").exists():
        subprocess.run(["git", "clone", "https://github.com/Duy-Thanh/ArtificialIntelligent.git"])
        os.chdir("ArtificialIntelligent")
    
    # Create necessary directories
    Path("Checkpoints").mkdir(exist_ok=True)
    Path("Datasets").mkdir(exist_ok=True)
    
    # Check if datasets are available
    dataset_files = [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
        "validation-00000-of-00001.parquet"
    ]
    
    datasets_present = all(
        Path("Datasets", filename).exists() 
        for filename in dataset_files
    )
    
    if not datasets_present:
        print("Please ensure the following files are in your Google Drive's Datasets folder:")
        for file in dataset_files:
            print(f"  - {file}")
        print("\nThen run the notebook again.")
    else:
        print("Dataset files found successfully!")
    
    print("Setup complete!")

if __name__ == "__main__":
    setup_colab() 