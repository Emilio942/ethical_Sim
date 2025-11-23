import unittest
import numpy as np
import os
import sys
import scipy.signal as signal

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.audio_processor import AudioLoader, AuditoryFeatures
from core.auditory_cortex import AuditoryExpertModule

class TestEngineerFixes(unittest.TestCase):
    
    def setUp(self):
        self.sr = 16000
        self.duration = 1.0
        self.t = np.linspace(0, self.duration, int(self.sr * self.duration))
        # Generate a chirp signal (sweep)
        self.audio = signal.chirp(self.t, f0=100, f1=8000, t1=self.duration, method='linear')
        self.loader = AudioLoader(target_sr=self.sr)
        self.features = AuditoryFeatures(sr=self.sr)
        
    def test_pre_emphasis(self):
        """Test if pre-emphasis boosts high frequencies."""
        # Create a signal with equal energy at low and high freq (white noise would be better but chirp is deterministic)
        # Let's use a simple step or just check the filter response on the chirp
        
        # Apply pre-emphasis
        pre_emphasized = self.loader.apply_pre_emphasis(self.audio)
        
        # Check energy at start (low freq) vs end (high freq)
        # In original chirp, amplitude is constant 1.0
        # In pre-emphasized, high freq should be higher amplitude than low freq (relative to original)
        # Actually, pre-emphasis is a high pass, so it attenuates low freqs more than high freqs.
        # Since chirp is constant amplitude, the output amplitude should increase over time (as freq increases).
        
        start_rms = np.sqrt(np.mean(pre_emphasized[:1000]**2))
        end_rms = np.sqrt(np.mean(pre_emphasized[-1000:]**2))
        
        print(f"Pre-emphasis: Low Freq RMS={start_rms:.4f}, High Freq RMS={end_rms:.4f}")
        self.assertGreater(end_rms, start_rms, "High frequencies should be boosted (or low attenuated) by pre-emphasis")

    def test_dilated_convolution(self):
        """Test if dilated convolution runs and produces correct shape."""
        # Generate dummy cochleogram (64 channels, 100 time steps)
        cochleogram = np.random.rand(64, 100).astype(np.float32)
        
        # Initialize module
        module = AuditoryExpertModule(module_id="test_module", best_freq_range=(0, 64))
        
        # Add filters manually
        # Kernel shape: (64, 5) - full height, 5 time steps
        kernel = np.random.rand(64, 5).astype(np.float32)
        module.add_filter(kernel)
        module.add_filter(kernel) # Add 2 filters
        
        # Run standard process
        standard_out = module.process(cochleogram)
        
        # Run dilated process
        dilated_out = module.process_dilated(cochleogram, dilation_rate=2)
        
        print(f"Standard Output Shape: {standard_out.shape}")
        print(f"Dilated Output Shape: {dilated_out.shape}")
        
        self.assertEqual(dilated_out.shape[0], 2)
        # Dilated kernel is wider, so valid output should be smaller in time dimension
        self.assertLess(dilated_out.shape[1], standard_out.shape[1], "Dilated output should be smaller due to larger effective kernel")

    def test_sonification(self):
        """Test reconstruction from cochleogram."""
        # Generate cochleogram
        cochleogram = self.features.compute_cochleogram(self.audio)
        
        # Reconstruct
        reconstructed = self.features.reconstruct_from_cochleogram(cochleogram)
        
        print(f"Original Duration: {len(self.audio)/self.sr:.2f}s")
        print(f"Reconstructed Duration: {len(reconstructed)/self.sr:.2f}s")
        
        # Check if not empty
        self.assertGreater(np.max(np.abs(reconstructed)), 0.0)
        
        # Lengths might differ slightly due to padding/downsampling
        diff = abs(len(self.audio) - len(reconstructed))
        self.assertLess(diff, 1000, "Reconstructed length should be close to original")

if __name__ == '__main__':
    unittest.main()
