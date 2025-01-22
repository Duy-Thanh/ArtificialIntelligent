import json
import pandas as pd
from typing import List, Union, Dict, Optional, Generator, Callable, Tuple
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, MarianMTModel, MarianTokenizer
import logging
import glob
import concurrent.futures
import hashlib
import pickle
import os
from tqdm import tqdm
import psutil
import numpy as np
import re
from html import unescape
import unicodedata
from random import random, choice, randint
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.tag import pos_tag
import torch
from config.data_config import DATASET_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update the NLTK downloads at the beginning of the file
try:
    # Download all required NLTK data
    nltk_resources = [
        'punkt',
        'wordnet',
        'averaged_perceptron_tagger',
        'stopwords',
        'punkt_tab'  # Add this missing resource
    ]
    
    for resource in nltk_resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.warning(f"Failed to download NLTK resource {resource}: {e}")
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")
    logger.warning("Some text augmentation features might not work properly")

class DataValidator:
    """Handles data validation and integrity checks"""
    
    @staticmethod
    def validate_text(text: str) -> bool:
        """Basic text validation"""
        if not isinstance(text, str):
            return False
        if not text.strip():
            return False
        return True
    
    @staticmethod
    def validate_file_integrity(file_path: str) -> bool:
        """Check if file is readable and not corrupted"""
        try:
            suffix = Path(file_path).suffix.lower()
            if suffix == '.parquet':
                # Just try to read the metadata
                pd.read_parquet(file_path, engine='pyarrow')
            elif suffix == '.csv':
                pd.read_csv(file_path, nrows=1)
            elif suffix in ['.json', '.jsonl']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if suffix == '.json':
                        json.load(f)
                    else:
                        next(f)
            return True
        except Exception as e:
            logger.error(f"File integrity check failed for {file_path}: {e}")
            return False

class MemoryEfficientLoader:
    """Handles memory-efficient data loading"""
    
    @staticmethod
    def get_chunk_size(file_size: int) -> int:
        """Calculate optimal chunk size based on available memory"""
        available_memory = psutil.virtual_memory().available
        chunk_size = max(1000, int(available_memory / (10 * file_size)))  # Use 10% of ratio
        return min(chunk_size, 100000)  # Cap at 100k rows
    
    @staticmethod
    def load_parquet_chunks(file_path: str, text_field: str) -> Generator[List[str], None, None]:
        """Load parquet file in chunks"""
        file_size = os.path.getsize(file_path)
        chunk_size = MemoryEfficientLoader.get_chunk_size(file_size)
        
        for chunk in pd.read_parquet(file_path, columns=[text_field], chunksize=chunk_size):
            yield chunk[text_field].dropna().tolist()

class TextDatasetLoader:
    """Handles loading text data from various file formats"""
    
    CACHE_DIR = Path(".cache/datasets")
    
    @classmethod
    def get_cache_path(cls, file_path: str, text_field: str) -> Path:
        """Generate cache path for a file"""
        file_hash = hashlib.md5(f"{file_path}:{text_field}".encode()).hexdigest()
        return cls.CACHE_DIR / f"{file_hash}.cache"

    @classmethod
    def load_from_cache(cls, file_path: str, text_field: str) -> Optional[List[str]]:
        """Try to load data from cache"""
        cache_path = cls.get_cache_path(file_path, text_field)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {file_path}: {e}")
        return None

    @classmethod
    def save_to_cache(cls, file_path: str, text_field: str, texts: List[str]):
        """Save loaded data to cache"""
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = cls.get_cache_path(file_path, text_field)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(texts, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {file_path}: {e}")

    @staticmethod
    def load_json(file_path: str, text_field: str) -> List[str]:
        """Load text data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [item[text_field] for item in data if text_field in item]
            elif isinstance(data, dict):
                return [data[text_field]] if text_field in data else []
    
    @staticmethod
    def load_jsonl(file_path: str, text_field: str) -> List[str]:
        """Load text data from JSONL file"""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if text_field in item:
                        texts.append(item[text_field])
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")
        return texts
    
    @staticmethod
    def load_csv(file_path: str, text_field: str) -> List[str]:
        """Load text data from CSV file"""
        df = pd.read_csv(file_path)
        if text_field not in df.columns:
            raise ValueError(f"Column '{text_field}' not found in CSV file")
        return df[text_field].dropna().tolist()
    
    @staticmethod
    def load_parquet(file_path: str, text_field: str) -> List[str]:
        """Load text data from Parquet file"""
        df = pd.read_parquet(file_path)
        if text_field not in df.columns:
            raise ValueError(f"Column '{text_field}' not found in Parquet file")
        return df[text_field].dropna().tolist()

    @staticmethod
    def get_files_from_pattern(pattern: Union[str, Path]) -> List[str]:
        """Get all files matching the pattern"""
        if isinstance(pattern, (str, Path)):
            pattern = str(pattern)
            # Handle both directory-based and sharded file patterns
            matched_files = glob.glob(pattern, recursive=True)
            return sorted(matched_files)  # Sort to ensure consistent order
        return []

    @staticmethod
    def is_sharded_pattern(pattern: Union[str, Path]) -> bool:
        """Check if the pattern represents sharded files"""
        pattern = str(pattern)
        return '*-of-*' in pattern

    @staticmethod
    def load_parquet_shard(args: tuple) -> List[str]:
        """Load a single parquet shard with validation"""
        file_path, text_field = args
        
        if not DataValidator.validate_file_integrity(file_path):
            raise ValueError(f"File integrity check failed: {file_path}")
        
        texts = []
        for chunk in MemoryEfficientLoader.load_parquet_chunks(file_path, text_field):
            valid_texts = [text for text in chunk if DataValidator.validate_text(text)]
            texts.extend(valid_texts)
            
        return texts

class TextPreprocessor:
    """Handles text preprocessing and cleaning operations"""
    
    def __init__(self, operations: List[Union[str, Callable]], **kwargs):
        """Initialize preprocessor with operations"""
        self.kwargs = kwargs
        self.funcs = []
        operation_map = {
            'basic_clean': self.basic_clean,
            'remove_urls': self.remove_urls,
            'remove_email': self.remove_email_addresses,
            'remove_special': self.remove_special_with_kwargs,
            'normalize_whitespace': self.normalize_whitespace
        }
        
        # Create list of functions to apply
        for op in operations:
            if isinstance(op, str):
                if op in operation_map:
                    self.funcs.append(operation_map[op])
                else:
                    logger.warning(f"Unknown operation: {op}")
            elif callable(op):
                self.funcs.append(op)
    
    def remove_special_with_kwargs(self, text: str) -> str:
        """Wrapper for remove_special_characters with kwargs"""
        keep_punctuation = self.kwargs.get('remove_special', {}).get('keep_punctuation', True)
        return self.remove_special_characters(text, keep_punctuation)
    
    def __call__(self, text: str) -> str:
        """Make the preprocessor callable"""
        for func in self.funcs:
            text = func(text)
        return text
    
    @staticmethod
    def basic_clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = unescape(text)
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    @staticmethod
    def remove_email_addresses(text: str) -> str:
        return re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    
    @staticmethod
    def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
        if keep_punctuation:
            pattern = r'[^a-zA-Z0-9\s.,!?-]'
        else:
            pattern = r'[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return ' '.join(text.split())

    @classmethod
    def create_pipeline(cls, operations: List[Union[str, Callable]], **kwargs) -> 'TextPreprocessor':
        """Create a preprocessing pipeline"""
        return cls(operations, **kwargs)

class ContextualAugmenter:
    """Handles contextual augmentations using pre-trained models"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.translation_models = {}
        self.translation_tokenizers = {}
        
    def load_translation_model(self, src_lang: str, tgt_lang: str):
        """Load translation model for a language pair"""
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        try:
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.translation_models[(src_lang, tgt_lang)] = model
            self.translation_tokenizers[(src_lang, tgt_lang)] = tokenizer
            return True
        except Exception as e:
            logger.warning(f"Failed to load translation model {model_name}: {e}")
            return False

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """Translate texts between languages"""
        if (src_lang, tgt_lang) not in self.translation_models:
            if not self.load_translation_model(src_lang, tgt_lang):
                return texts
        
        model = self.translation_models[(src_lang, tgt_lang)]
        tokenizer = self.translation_tokenizers[(src_lang, tgt_lang)]
        
        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = model.generate(**inputs)
            return tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return texts

class TextAugmenter:
    """Enhanced text augmentation with contextual operations"""
    
    def __init__(self, p: float = 0.3):
        self.p = p
        self.contextual_augmenter = ContextualAugmenter()
        self.stopwords = set(stopwords.words('english'))
    
    @staticmethod
    def get_synonyms(word: str) -> List[str]:
        """Get synonyms for a word using WordNet"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.append(lemma.name())
        return list(set(synonyms))
    
    def synonym_replacement(self, text: str) -> str:
        """Replace random words with their synonyms"""
        if random() > self.p:
            return text
            
        words = word_tokenize(text)
        new_words = words.copy()
        
        n_to_replace = max(1, int(len(words) * 0.1))  # Replace up to 10% of words
        for _ in range(n_to_replace):
            idx = randint(0, len(words) - 1)
            synonyms = self.get_synonyms(words[idx])
            if synonyms:
                new_words[idx] = choice(synonyms)
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str) -> str:
        """Randomly delete words"""
        if random() > self.p:
            return text
            
        words = word_tokenize(text)
        if len(words) <= 3:  # Keep very short sentences intact
            return text
            
        new_words = []
        for word in words:
            if random() > 0.1:  # 10% chance to delete each word
                new_words.append(word)
                
        if not new_words:  # Ensure we don't return empty text
            new_words = [choice(words)]
            
        return ' '.join(new_words)
    
    def random_swap(self, text: str) -> str:
        """Randomly swap two words"""
        if random() > self.p:
            return text
            
        words = word_tokenize(text)
        if len(words) <= 3:
            return text
            
        for _ in range(max(1, int(len(words) * 0.1))):  # Swap up to 10% of words
            idx1, idx2 = randint(0, len(words) - 1), randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def back_translation(self, text: str) -> str:
        """Simulate back translation by introducing minor variations"""
        if random() > self.p:
            return text
            
        # Simple simulation of translation artifacts
        variations = [
            lambda x: x.lower(),
            lambda x: x.replace('the ', 'a '),
            lambda x: x.replace('is ', 'was '),
            lambda x: x.replace('are ', 'were '),
        ]
        
        return choice(variations)(text)
    
    def advanced_back_translation(self, text: str) -> str:
        """Back translation through multiple languages"""
        if random() > self.p:
            return text
            
        # Define translation chain
        language_chains = [
            ['en', 'fr', 'en'],
            ['en', 'de', 'en'],
            ['en', 'es', 'en'],
            ['en', 'ru', 'en']
        ]
        
        chain = choice(language_chains)
        current_text = text
        
        try:
            for i in range(len(chain) - 1):
                src_lang, tgt_lang = chain[i], chain[i + 1]
                translated = self.contextual_augmenter.translate([current_text], src_lang, tgt_lang)[0]
                current_text = translated
            return current_text
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text
    
    def contextual_word_replacement(self, text: str) -> str:
        """Replace words based on their POS tags and context"""
        if random() > self.p:
            return text
            
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        
        new_words = words.copy()
        n_to_replace = max(1, int(len(words) * 0.15))  # Replace up to 15% of words
        
        for _ in range(n_to_replace):
            idx = randint(0, len(words) - 1)
            word, pos = pos_tags[idx]
            
            # Skip stopwords and short words
            if word.lower() in self.stopwords or len(word) < 4:
                continue
            
            # Get context-appropriate replacements
            if pos.startswith('NN'):  # Nouns
                synonyms = self.get_synonyms(word)
            elif pos.startswith('VB'):  # Verbs
                synonyms = self.get_verb_forms(word)
            elif pos.startswith('JJ'):  # Adjectives
                synonyms = self.get_synonyms(word)
            else:
                continue
                
            if synonyms:
                new_words[idx] = choice(synonyms)
        
        return ' '.join(new_words)
    
    def get_verb_forms(self, verb: str) -> List[str]:
        """Get different forms of a verb"""
        forms = set()
        for synset in wordnet.synsets(verb, pos=wordnet.VERB):
            for lemma in synset.lemmas():
                forms.add(lemma.name())
        return list(forms)
    
    def create_augmentation_pipeline(self, operations: List[str] = None) -> Callable:
        """Create enhanced augmentation pipeline"""
        if operations is None:
            operations = [
                'synonym_replacement',
                'contextual_word_replacement',
                'advanced_back_translation',
                'random_swap'
            ]
            
        op_map = {
            'synonym_replacement': self.synonym_replacement,
            'random_deletion': self.random_deletion,
            'random_swap': self.random_swap,
            'back_translation': self.back_translation,
            'advanced_back_translation': self.advanced_back_translation,
            'contextual_word_replacement': self.contextual_word_replacement
        }
        
        available_ops = [op_map[op] for op in operations if op in op_map]
        
        def pipeline(text: str) -> List[str]:
            augmented_texts = [text]  # Always include original
            for op in available_ops:
                augmented = op(text)
                if augmented != text:
                    augmented_texts.append(augmented)
            return list(set(augmented_texts))  # Remove duplicates
            
        return pipeline

class TextDataset(Dataset):
    """Dataset class that handles multiple file formats and preprocessing"""
    
    def __init__(
        self,
        file_path: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        text_field: str = "text",
        max_length: int = 512,
        recursive: bool = True,
        validate_data: bool = True,
        memory_efficient: bool = True,
        use_cache: bool = True,
        preprocessing_pipeline: Optional[Union[List[str], Callable]] = None,
        preprocessing_kwargs: Dict = None,
        augmentation_pipeline: Optional[List[str]] = None,
        augmentation_p: float = 0.3,
        batch_size: int = 100,
        **kwargs
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validate_data = validate_data
        self.memory_efficient = memory_efficient
        self.use_cache = use_cache
        
        # Set up preprocessing
        if isinstance(preprocessing_pipeline, list):
            self.preprocessing_pipeline = TextPreprocessor.create_pipeline(
                preprocessing_pipeline,
                **(preprocessing_kwargs or {})
            )
        else:
            self.preprocessing_pipeline = preprocessing_pipeline
            
        # Set up augmentation
        self.augmentation_pipeline = None
        if augmentation_pipeline:
            augmenter = TextAugmenter(p=augmentation_p)
            self.augmentation_pipeline = augmenter.create_augmentation_pipeline(augmentation_pipeline)
        
        # Load texts
        self.texts = self._load_data(file_path, text_field)
        
        # Initialize lazy loading attributes
        self._batch_size = batch_size
        self._cached_encodings = {}
        self._max_cache_size = 5  # Keep at most 5 batches in memory
        
    def _load_data(
        self,
        file_path: Union[str, List[str]],
        text_field: str,
        file_type: str = None
    ) -> List[str]:
        """Load data from file(s) or directory based on format"""
        
        if isinstance(file_path, (str, Path)):
            path = Path(file_path)
            if path.is_dir():
                # Directory handling
                patterns = [f"**/*.{ext}" if self.recursive else f"*.{ext}"
                          for ext in ['json', 'jsonl', 'csv', 'parquet']]
                all_files = []
                for pattern in patterns:
                    all_files.extend(list(path.glob(pattern)))
                return self._load_multiple_files(all_files, text_field, file_type)
            else:
                # Handle both sharded and regular file patterns
                matching_files = TextDatasetLoader.get_files_from_pattern(path)
                if matching_files:
                    return self._load_multiple_files(matching_files, text_field, file_type)
                else:
                    return self._load_single_file(str(path), text_field, file_type)
        elif isinstance(file_path, list):
            # Handle list of patterns/paths
            all_files = []
            for pattern in file_path:
                matched_files = TextDatasetLoader.get_files_from_pattern(pattern)
                all_files.extend(matched_files)
            return self._load_multiple_files(sorted(set(all_files)), text_field, file_type)
        else:
            raise ValueError(f"Unsupported file_path type: {type(file_path)}")
    
    def _load_multiple_files(
        self,
        file_paths: List[Union[str, Path]],
        text_field: str,
        file_type: str = None
    ) -> List[str]:
        """Load data from multiple files with parallel processing and validation"""
        all_texts = []
        
        if not file_paths:
            raise ValueError("No valid files found to load")
        
        # Validate files first if needed
        if self.validate_data:
            invalid_files = []
            for fp in file_paths:
                if not DataValidator.validate_file_integrity(str(fp)):
                    invalid_files.append(fp)
            
            if invalid_files:
                logger.warning(f"Skipping {len(invalid_files)} invalid files")
                file_paths = [fp for fp in file_paths if fp not in invalid_files]
        
        # Group files by type
        file_groups = {}
        for fp in file_paths:
            ft = file_type or Path(fp).suffix.lower()[1:]
            if ft not in file_groups:
                file_groups[ft] = []
            file_groups[ft].append(str(fp))

        # Process each file type
        for ft, fps in file_groups.items():
            if TextDatasetLoader.is_sharded_pattern(fps[0]) and ft == 'parquet':
                # Parallel loading for sharded parquet files
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_file = {
                        executor.submit(TextDatasetLoader.load_parquet_shard, (fp, text_field)): fp 
                        for fp in fps
                    }
                    
                    with tqdm(total=len(fps), desc="Loading shards") as pbar:
                        for future in concurrent.futures.as_completed(future_to_file):
                            file_path = future_to_file[future]
                            try:
                                texts = future.result()
                                all_texts.extend(texts)
                                logger.info(f"Successfully loaded {len(texts)} samples from {file_path}")
                            except Exception as e:
                                logger.error(f"Error loading file {file_path}: {str(e)}")
                            pbar.update(1)
            else:
                # Sequential loading for other file types
                for fp in tqdm(fps, desc=f"Loading {ft} files"):
                    try:
                        if self.use_cache and not self.memory_efficient:
                            texts = TextDatasetLoader.load_from_cache(fp, text_field)
                            if texts is None:
                                texts = self._load_single_file(fp, text_field, ft)
                                TextDatasetLoader.save_to_cache(fp, text_field, texts)
                        else:
                            texts = self._load_single_file(fp, text_field, ft)
                        
                        if self.validate_data:
                            texts = [t for t in texts if DataValidator.validate_text(t)]
                            
                        all_texts.extend(texts)
                        logger.info(f"Successfully loaded {len(texts)} samples from {fp}")
                    except Exception as e:
                        logger.error(f"Error loading file {fp}: {str(e)}")

        # Add preprocessing step after loading texts
        if all_texts and self.preprocessing_pipeline is not None:
            logger.info("Applying preprocessing pipeline...")
            all_texts = self._preprocess_texts(all_texts)
            logger.info(f"Preprocessing complete. {len(all_texts)} texts remaining.")
        
        if not all_texts:
            raise ValueError("No valid texts loaded from the files")
        
        return all_texts
    
    def _load_single_file(
        self,
        file_path: str,
        text_field: str,
        file_type: str = None
    ) -> List[str]:
        """Load data from a single file"""
        
        if file_type is None:
            file_type = Path(file_path).suffix.lower()[1:]
        
        loader_map = {
            'json': TextDatasetLoader.load_json,
            'jsonl': TextDatasetLoader.load_jsonl,
            'csv': TextDatasetLoader.load_csv,
            'parquet': TextDatasetLoader.load_parquet
        }
        
        if file_type not in loader_map:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        logger.info(f"Loading data from {file_path}")
        texts = loader_map[file_type](file_path, text_field)
        logger.info(f"Loaded {len(texts)} text samples")
        return texts
    
    def _encode_batch(self, batch_texts: List[str]) -> Dict:
        """Encode a batch of texts with proper padding"""
        if not batch_texts:
            return self._get_default_item()
        
        encodings = self.tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Ensure all tensors are padded to max_length
        for key in encodings:
            if isinstance(encodings[key], torch.Tensor):
                if encodings[key].size(1) < self.max_length:
                    pad_size = self.max_length - encodings[key].size(1)
                    pad_value = 0 if key != 'input_ids' else self.tokenizer.pad_token_id
                    encodings[key] = torch.nn.functional.pad(
                        encodings[key], 
                        (0, pad_size), 
                        value=pad_value
                    )
        
        return encodings

    def _get_default_item(self) -> Dict:
        """Create a default item with proper padding"""
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        input_ids = torch.full((self.max_length,), pad_token_id, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

    def __getitem__(self, idx: int) -> Dict:
        """Get a single item from the dataset"""
        # Calculate which batch this index belongs to
        batch_idx = idx // self._batch_size
        local_idx = idx % self._batch_size
        
        # Calculate batch slice
        start_idx = batch_idx * self._batch_size
        end_idx = min(start_idx + self._batch_size, len(self.texts))
        
        # If this is a new batch or the batch isn't cached
        if batch_idx not in self._cached_encodings:
            # Encode batch
            batch_texts = self.texts[start_idx:end_idx]
            self._cached_encodings[batch_idx] = self._encode_batch(batch_texts)
            
            # Clean up old cache entries
            current_batch_keys = {
                max(0, batch_idx - 1),
                batch_idx,
                batch_idx + 1
            }
            # Keep only adjacent batches
            keys_to_remove = [k for k in self._cached_encodings.keys() 
                             if k not in current_batch_keys]
            for key in keys_to_remove:
                del self._cached_encodings[key]
        
        try:
            # Get encodings for this item
            batch_encodings = self._cached_encodings[batch_idx]
            if local_idx >= len(batch_encodings['input_ids']):
                # Handle edge case where local_idx is out of bounds
                local_idx = len(batch_encodings['input_ids']) - 1
            
            item = {
                key: val[local_idx] 
                for key, val in batch_encodings.items()
            }
            item['labels'] = item['input_ids'].clone()
            return item
        except Exception as e:
            logger.error(f"Error getting item {idx} (batch {batch_idx}, local {local_idx}): {str(e)}")
            # Return a default item with proper padding if there's an error
            return self._get_default_item()
    
    def __len__(self) -> int:
        """Get the total size of the dataset"""
        return len(self.texts)

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Apply preprocessing and augmentation pipeline to texts"""
        processed_texts = []
        
        for text in tqdm(texts, desc="Processing texts"):
            try:
                # Apply preprocessing
                processed_text = text
                if self.preprocessing_pipeline:
                    processed_text = self.preprocessing_pipeline(text)
                    if not processed_text or not processed_text.strip():
                        continue
                
                processed_texts.append(processed_text)
                    
            except Exception as e:
                logger.debug(f"Error processing text: {str(e)}")  # Changed to debug level
                processed_texts.append(text)
        
        # Apply augmentation after preprocessing if enabled
        if self.augmentation_pipeline and processed_texts:
            augmented_texts = []
            for text in processed_texts:
                try:
                    augmented = self.augmentation_pipeline(text)
                    if augmented:
                        augmented_texts.extend(augmented)
                except Exception as e:
                    logger.debug(f"Augmentation failed: {str(e)}")  # Changed to debug level
            
            if augmented_texts:
                processed_texts.extend(augmented_texts)
        
        # Ensure we have at least some texts
        if not processed_texts:
            logger.warning("No texts remained after processing, keeping originals")
            return texts
        
        return processed_texts

# Example usage functions
def load_and_combine_datasets(
    file_paths: List[str],
    tokenizer: PreTrainedTokenizer,
    text_field: str = "text",
    **kwargs
) -> TextDataset:
    """Helper function to load and combine multiple datasets"""
    return TextDataset(file_paths, tokenizer, text_field, **kwargs)

def get_dataset_info(dataset: TextDataset) -> Dict:
    """Helper function to get dataset statistics"""
    return {
        "total_samples": len(dataset),
        "vocab_size": len(dataset.tokenizer),
        "max_length": dataset.max_length
    }

def load_dataset_split(split: str, tokenizer: PreTrainedTokenizer, **kwargs) -> Dataset:
    """Load a dataset split with preprocessing"""
    # Get base directory and split config
    base_dir = Path(os.environ.get('DATASET_DIR', 'Datasets')).resolve()  # Use environment variable
    logger.info(f"Looking for {split} dataset in: {base_dir}")
    
    split_config = DATASET_CONFIG.get(split)
    if not split_config:
        raise ValueError(f"No configuration found for split: {split}")
    
    # Find files directly using glob patterns
    file_patterns = [
        str(base_dir / f"{split}-*-of-*.parquet"),  # Only look for files matching the split
    ]
    
    logger.info(f"Searching for files with patterns: {file_patterns}")
    
    # Find matching files
    matching_files = []
    for pattern in file_patterns:
        matches = glob.glob(pattern)
        logger.info(f"Pattern {pattern} matched files: {matches}")
        matching_files.extend(matches)
    
    if not matching_files:
        # Try alternative paths
        alt_base_dir = Path("../Datasets").resolve()
        logger.info(f"Trying alternative path: {alt_base_dir}")
        
        alt_patterns = [
            str(alt_base_dir / f"{split}-*-of-*.parquet"),
        ]
        
        for pattern in alt_patterns:
            matches = glob.glob(pattern)
            logger.info(f"Alt pattern {pattern} matched files: {matches}")
            matching_files.extend(matches)
    
    if not matching_files:
        logger.error(f"Current directory: {Path.cwd()}")
        logger.error(f"Directory contents: {list(Path.cwd().glob('*'))}")
        logger.error(f"Datasets directory contents: {list(base_dir.glob('*'))}")
        raise ValueError(f"No files found for split {split} in {base_dir}")
    
    logger.info(f"Found {len(matching_files)} files: {matching_files}")
    file_paths = sorted(matching_files)
    
    # Load and preprocess dataset
    return TextDataset(
        file_paths,
        tokenizer=tokenizer,
        text_field=split_config.get("text_field", "text"),
        max_length=split_config.get("max_length", 512),
        **kwargs
    ) 