import unittest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.audio_processor import AuditoryFeatures
from core.auditory_cortex import AuditoryCortex, STRFGenerator, LIFNeuron, AuditoryExpertModule

class TestAuditoryCortex(unittest.TestCase):
    def setUp(self):
        self.features = AuditoryFeatures(sr=16000, num_channels=64)
        self.generator = STRFGenerator(sr=16000, num_channels=64, time_steps=20)
        self.cortex = AuditoryCortex(num_modules=8, num_channels=64)
        self.cortex.initialize_filters(self.generator)

    def test_pure_tone_response(self):
        """
        Test if modules respond to their best frequency (Tonotopy).
        """
        # Generate pure tone at different frequencies
        duration = 0.5
        t = np.linspace(0, duration, int(16000 * duration))
        
        # Test Low Frequency (e.g., Module 0)
        # Module 0 covers channels 0-8. 
        # Channel 0 is 100Hz. Channel 63 is 8000Hz.
        # Let's pick a low freq.
        freq_low = 150 # Hz
        tone_low = np.sin(2 * np.pi * freq_low * t).astype(np.float32)
        
        # Test High Frequency (e.g., Module 7)
        freq_high = 6000 # Hz
        tone_high = np.sin(2 * np.pi * freq_high * t).astype(np.float32)
        
        # Process
        # compute_cochleogram expects 1D array
        coch_low = self.features.compute_cochleogram(tone_low)
        coch_high = self.features.compute_cochleogram(tone_high)
        
        resp_low = self.cortex.process_input(coch_low)
        resp_high = self.cortex.process_input(coch_high)
        
        # Check activations
        # Module 0 should be active for low tone
        # Module 7 should be active for high tone
        
        # Sum of activations (energy)
        act_low_mod0 = np.sum(np.abs(resp_low['A1_Module_0']))
        act_low_mod7 = np.sum(np.abs(resp_low['A1_Module_7']))
        
        act_high_mod0 = np.sum(np.abs(resp_high['A1_Module_0']))
        act_high_mod7 = np.sum(np.abs(resp_high['A1_Module_7']))
        
        print(f"Low Tone (150Hz): Mod0 (Low)={act_low_mod0:.2f}, Mod7 (High)={act_low_mod7:.2f}")
        print(f"High Tone (6000Hz): Mod0 (Low)={act_high_mod0:.2f}, Mod7 (High)={act_high_mod7:.2f}")
        
        # Assert Tonotopy
        self.assertGreater(act_low_mod0, act_low_mod7, "Low frequency tone should activate low-freq module more")
        self.assertGreater(act_high_mod7, act_high_mod0, "High frequency tone should activate high-freq module more")

    def test_modulation_transfer_function(self):
        """
        Test Modulation Transfer Function (MTF).
        Verifies that modules/filters tuned to specific temporal modulations 
        respond best to matching AM rates.
        Implements Phase 4.1 (MTF-Test).
        """
        # Create a specific filter tuned to 10Hz modulation
        # We'll use a specific module for testing
        module = self.cortex.modules[4] # Middle frequency module
        module.filters = [] # Clear existing
        
        # Generate STRF tuned to 10Hz temporal modulation
        # spectral_mod_scale=0 means no spectral modulation (broadband-ish)
        bf_idx = (module.best_freq_range[0] + module.best_freq_range[1]) // 2
        target_rate = 10.0 # Hz
        strf_10hz = self.generator.generate_gabor_strf(
            best_freq_idx=bf_idx, 
            spectral_width=10.0, 
            temporal_mod_rate=target_rate, 
            spectral_mod_scale=0.0,
            direction=1
        )
        module.add_filter(strf_10hz)
        
        # Generate AM Noise stimuli at different rates
        duration = 1.0
        t = np.linspace(0, duration, int(16000 * duration))
        carrier = np.random.randn(len(t)) # White noise
        
        rates = [2.0, 10.0, 30.0] # Hz
        responses = {}
        
        for rate in rates:
            # AM Noise: (1 + m * sin(2*pi*f*t)) * carrier
            modulator = 1.0 + 0.8 * np.sin(2 * np.pi * rate * t)
            stimulus = carrier * modulator
            stimulus = stimulus.astype(np.float32)
            
            # Process
            coch = self.features.compute_cochleogram(stimulus)
            
            # We only care about the specific module we configured
            # The module process returns (num_filters, time_steps)
            # We take the mean activation (energy)
            activation = module.process(coch)
            energy = np.mean(np.abs(activation))
            responses[rate] = energy
            
        print(f"MTF Responses for 10Hz Filter: {responses}")
        
        # Assert that 10Hz stimulus produces highest response
        self.assertGreater(responses[10.0], responses[2.0], "Filter should prefer 10Hz over 2Hz")
        self.assertGreater(responses[10.0], responses[30.0], "Filter should prefer 10Hz over 30Hz")

    def test_spiking_dynamics(self):
        """
        Test Spiking Dynamics (LIF Neuron).
        Verifies that stronger inputs cause more spikes.
        We bypass PCEN here to ensure we are testing the neuron's sensitivity to input magnitude directly.
        """
        # Create a module with spiking enabled
        spiking_module = AuditoryExpertModule("Spiking_Test", (0, 64))
        
        # Add a simple broadband filter (impulse)
        simple_filter = np.zeros((64, 20))
        simple_filter[:, 10] = 1.0 # Active at t=10
        spiking_module.add_filter(simple_filter)
        spiking_module.use_spiking = True
        spiking_module.neurons = [LIFNeuron(threshold=0.5, v_reset=0.0)]
        
        # 1. Strong Input (Synthetic Cochleogram)
        # Shape: (64, 100)
        coch_strong = np.zeros((64, 100))
        # Add a pulse that matches the filter
        coch_strong[:, 40:60] = 10.0 # High amplitude
        
        resp_strong = spiking_module.process(coch_strong)
        spike_count_strong = np.sum(resp_strong)
        
        # 2. Weak Input (Synthetic Cochleogram)
        # Reset neurons for fair comparison (or create new module)
        spiking_module.neurons = [LIFNeuron(threshold=0.5, v_reset=0.0)]
        
        coch_weak = np.zeros((64, 100))
        coch_weak[:, 40:60] = 0.1 # Low amplitude
        
        resp_weak = spiking_module.process(coch_weak)
        spike_count_weak = np.sum(resp_weak)
        
        print(f"Spikes (Strong Input): {spike_count_strong}")
        print(f"Spikes (Weak Input): {spike_count_weak}")
        
        self.assertGreater(spike_count_strong, spike_count_weak, "Strong input should cause more spikes")
        self.assertTrue(np.all(np.isin(resp_strong, [0, 1])), "Output should be binary spikes")

if __name__ == '__main__':
    unittest.main()
