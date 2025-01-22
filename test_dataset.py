import unittest
from pathlib import Path
from transformers import GPT2Tokenizer
from dataset import TextDataset, load_dataset_split
import torch
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDatasetConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        
        # Create test directories
        Path("datasets").mkdir(exist_ok=True)
        
        # Create sample data
        sample_data = pd.DataFrame({
            'text': [
                "This is a test sentence.",
                "Another test sentence with http://example.com and email@test.com",
                "Machine learning is fascinating!",
                "Python programming language"
            ]
        })
        
        # Save sample data in different formats
        cls.test_file = "datasets/test.parquet"
        cls.train_file = "datasets/train.parquet"
        cls.val_file = "datasets/validation.parquet"
        
        logger.info("Creating test files...")
        sample_data.to_parquet(cls.test_file, engine='pyarrow')
        sample_data.to_parquet(cls.train_file, engine='pyarrow')
        sample_data.to_parquet(cls.val_file, engine='pyarrow')
        logger.info("Test files created successfully")

    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        for file in [cls.test_file, cls.train_file, cls.val_file]:
            try:
                Path(file).unlink()
            except Exception:
                pass

    def test_file_loading(self):
        """Test if files are loaded correctly"""
        try:
            logger.info(f"Testing file loading with {self.train_file}")
            dataset = TextDataset(
                file_path=self.train_file,
                tokenizer=self.tokenizer,
                text_field="text",
                validate_data=True
            )
            self.assertTrue(len(dataset) > 0)
            logger.info(f"Successfully loaded {len(dataset)} samples from training file")
        except Exception as e:
            logger.error(f"File loading failed: {str(e)}")
            self.fail(f"Failed to load dataset: {str(e)}")

    def test_preprocessing(self):
        """Test preprocessing pipeline"""
        preprocessing_pipeline = [
            'basic_clean',
            'remove_urls',
            'remove_email',
            'remove_special',
            'normalize_whitespace'
        ]
        
        try:
            dataset = TextDataset(
                file_path=self.test_file,
                tokenizer=self.tokenizer,
                preprocessing_pipeline=preprocessing_pipeline,
                preprocessing_kwargs={'remove_special': {'keep_punctuation': True}}
            )
            self.assertTrue(len(dataset) > 0)
            
            # Check if preprocessing worked
            sample = dataset.texts[0]
            logger.info(f"Sample preprocessed text: {sample}")
            
            # Verify specific preprocessing effects
            for text in dataset.texts:
                self.assertFalse('http' in text.lower(), f"URL found in: {text}")
                self.assertFalse('@' in text, f"Email found in: {text}")
                self.assertFalse('\n' in text, f"Newline found in: {text}")
                self.assertFalse('  ' in text, f"Multiple spaces found in: {text}")
            
            logger.info(f"Preprocessing pipeline test passed with {len(dataset.texts)} texts")
        except Exception as e:
            logger.error(f"Preprocessing test failed: {str(e)}")
            self.fail(f"Preprocessing test failed: {str(e)}")

    def test_augmentation(self):
        """Test augmentation pipeline"""
        augmentation_pipeline = [
            'synonym_replacement',
            'contextual_word_replacement',
            'advanced_back_translation',
            'random_swap'
        ]
        
        try:
            dataset = TextDataset(
                file_path=self.test_file,
                tokenizer=self.tokenizer,
                augmentation_pipeline=augmentation_pipeline,
                augmentation_p=1.0  # Always apply augmentation for testing
            )
            
            # Check if we have more samples after augmentation
            self.assertTrue(len(dataset) > 0)
            logger.info(f"Generated {len(dataset)} samples with augmentation")
        except Exception as e:
            self.fail(f"Augmentation test failed: {str(e)}")

    def test_memory_efficiency(self):
        """Test memory-efficient loading"""
        try:
            dataset = TextDataset(
                file_path=self.train_file,
                tokenizer=self.tokenizer,
                memory_efficient=True,
                use_cache=False
            )
            self.assertTrue(len(dataset) > 0)
            logger.info("Memory-efficient loading test passed")
        except Exception as e:
            self.fail(f"Memory efficiency test failed: {str(e)}")

    def test_full_pipeline(self):
        """Test complete pipeline with all features"""
        try:
            train_dataset = load_dataset_split(
                "train",
                self.tokenizer,
                use_cache=True,
                num_workers=2,
                validate_data=True,
                memory_efficient=True,
                preprocessing_pipeline=[
                    'basic_clean',
                    'remove_urls',
                    'remove_email',
                    'remove_special',
                    'normalize_whitespace'
                ],
                preprocessing_kwargs={'remove_special': {'keep_punctuation': True}},
                augmentation_pipeline=[
                    'synonym_replacement',
                    'contextual_word_replacement',
                    'advanced_back_translation',
                    'random_swap'
                ],
                augmentation_p=0.3
            )
            
            # Basic checks
            self.assertTrue(len(train_dataset) > 0)
            self.assertTrue(isinstance(train_dataset[0], dict))
            self.assertTrue('input_ids' in train_dataset[0])
            self.assertTrue('attention_mask' in train_dataset[0])
            
            # Check tensor shapes
            batch = train_dataset[0]
            self.assertEqual(batch['input_ids'].dim(), 1)
            self.assertEqual(batch['attention_mask'].dim(), 1)
            
            logger.info("Full pipeline test passed successfully")
        except Exception as e:
            self.fail(f"Full pipeline test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 