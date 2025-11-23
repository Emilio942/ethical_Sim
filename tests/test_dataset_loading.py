import unittest
import os
import sys
import shutil

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.dataset_pipeline import DatasetPipeline

class TestDatasetLoading(unittest.TestCase):
    
    def setUp(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/emodb'))
        # Ensure dataset exists (it should be downloaded by now)
        if not os.path.exists(self.base_dir):
            self.skipTest("EMO-DB dataset not found. Run scripts/download_datasets.py first.")
            
        self.pipeline = DatasetPipeline(dataset_path=self.base_dir)
        
    def test_scan_dataset(self):
        """Test if the pipeline finds the files."""
        self.pipeline.scan_dataset()
        print(f"Found {len(self.pipeline.file_list)} files.")
        self.assertGreater(len(self.pipeline.file_list), 0, "Should find audio files")
        self.assertEqual(len(self.pipeline.file_list), 535, "Should find exactly 535 files for EMO-DB")
        
    def test_labels(self):
        """Test if labels are correctly extracted from folder names."""
        self.pipeline.scan_dataset()
        
        # Check a few labels
        labels = set(self.pipeline.labels.values())
        expected_labels = {'anger', 'boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral'}
        
        print(f"Found labels: {labels}")
        self.assertTrue(expected_labels.issubset(labels), "All emotion labels should be present")

    def test_batch_loading(self):
        """Test loading a batch of audio."""
        self.pipeline.scan_dataset()
        audio_batch, label_batch = self.pipeline.get_batch(batch_size=4)
        
        self.assertEqual(len(audio_batch), 4)
        self.assertEqual(len(label_batch), 4)
        
        # Check audio shape
        print(f"Audio shape: {audio_batch[0].shape}")
        self.assertGreater(len(audio_batch[0]), 0)

if __name__ == '__main__':
    unittest.main()
