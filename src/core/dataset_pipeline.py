import os
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from core.logger import logger
from core.audio_processor import AudioLoader

class DatasetPipeline:
    """
    Manages the loading and preparation of audio datasets for training/testing.
    Implements Phase 0.2 of the migration plan.
    """
    def __init__(self, dataset_path: str, target_sr: int = 16000):
        self.dataset_path = dataset_path
        self.loader = AudioLoader(target_sr=target_sr)
        self.file_list: List[str] = []
        self.labels: Dict[str, str] = {} # filename -> label
        
    def scan_dataset(self, file_extension: str = ".wav"):
        """
        Scans the dataset directory for audio files.
        Assumes a structure like: dataset_path/class_name/file.wav
        """
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset path not found: {self.dataset_path}")
            return

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(file_extension):
                    full_path = os.path.join(root, file)
                    self.file_list.append(full_path)
                    
                    # Extract label from folder name
                    label = os.path.basename(root)
                    self.labels[full_path] = label
                    
        logger.info(f"Found {len(self.file_list)} audio files in {self.dataset_path}")

    def get_batch(self, batch_size: int = 32) -> Tuple[List[np.ndarray], List[str]]:
        """
        Returns a random batch of audio data and labels.
        """
        if not self.file_list:
            return [], []
            
        batch_files = random.sample(self.file_list, min(batch_size, len(self.file_list)))
        audio_batch = []
        label_batch = []
        
        for file_path in batch_files:
            try:
                audio, _ = self.loader.load_audio(file_path)
                
                # Simple padding/truncating to fixed length (e.g. 1 sec)
                target_len = 16000 # 1 sec at 16kHz
                if len(audio) > target_len:
                    start = random.randint(0, len(audio) - target_len)
                    audio = audio[start:start+target_len]
                elif len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                    
                audio_batch.append(audio)
                label_batch.append(self.labels[file_path])
                
            except Exception as e:
                logger.error(f"Error loading batch file {file_path}: {e}")
                
        return audio_batch, label_batch

    def split_train_test(self, test_ratio: float = 0.2):
        """
        Splits the file list into train and test sets.
        (Simple implementation, just shuffles list)
        """
        random.shuffle(self.file_list)
        split_idx = int(len(self.file_list) * (1 - test_ratio))
        
        self.train_files = self.file_list[:split_idx]
        self.test_files = self.file_list[split_idx:]
        
        logger.info(f"Split dataset: {len(self.train_files)} train, {len(self.test_files)} test")
