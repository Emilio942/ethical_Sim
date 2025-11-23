import numpy as np
import scipy.signal as signal
from typing import Tuple, List, Dict, Optional
from core.logger import logger

class STRFGenerator:
    """
    Generates Spectro-Temporal Receptive Fields (STRFs) for auditory neurons.
    Implements Phase 2.2 of the migration plan.
    """
    def __init__(self, sr: int = 16000, num_channels: int = 64, time_steps: int = 20):
        self.sr = sr
        self.num_channels = num_channels
        self.time_steps = time_steps # Temporal width of the kernel (e.g., 20 steps * 10ms = 200ms)

    def generate_gabor_strf(self, 
                          best_freq_idx: int, 
                          spectral_width: float, 
                          temporal_mod_rate: float, 
                          spectral_mod_scale: float,
                          direction: int = 1) -> np.ndarray:
        """
        Generates a Gabor-like STRF kernel.
        
        Args:
            best_freq_idx: Center frequency index (channel).
            spectral_width: Width in channels.
            temporal_mod_rate: Temporal modulation rate (Hz).
            spectral_mod_scale: Spectral modulation scale (cycles/channel).
            direction: 1 for upward sweep, -1 for downward.
            
        Returns:
            kernel (np.ndarray): Shape (channels, time_steps)
        """
        # Create grid
        t = np.linspace(0, 0.2, self.time_steps) # 200ms window
        f = np.arange(self.num_channels)
        
        T, F = np.meshgrid(t, f)
        
        # Gaussian envelope
        f_center = best_freq_idx
        t_center = 0.1 # Center in time
        
        # Avoid division by zero
        spectral_width = max(spectral_width, 0.1)
        
        envelope = np.exp(-0.5 * ((F - f_center)**2 / spectral_width**2 + (T - t_center)**2 / 0.05**2))
        
        # Carrier (Gabor function)
        # Directionality is implemented by coupling time and frequency in the phase
        phase = 2 * np.pi * (temporal_mod_rate * T + direction * spectral_mod_scale * F)
        
        strf = envelope * np.cos(phase)
        
        # Normalize
        strf = strf / (np.max(np.abs(strf)) + 1e-8)
        
        return strf

class AuditoryExpertModule:
    """
    Simulates a cortical column / expert module in A1.
    Implements Phase 3.1 (Asymmetric 2D-CNNs simulation).
    """
    def __init__(self, module_id: str, best_freq_range: Tuple[int, int], use_spiking: bool = False):
        self.module_id = module_id
        self.best_freq_range = best_freq_range # Range of channels this module covers
        self.filters = []
        self.use_spiking = use_spiking
        self.neurons = [] # List of LIFNeuron instances, one per filter
        
    def add_filter(self, kernel: np.ndarray):
        self.filters.append(kernel)
        if self.use_spiking:
            self.neurons.append(LIFNeuron())
        
    def process(self, cochleogram: np.ndarray) -> np.ndarray:
        """
        Processes the input cochleogram with the module's filters.
        
        Args:
            cochleogram: Shape (channels, time_steps)
            
        Returns:
            activations: Shape (num_filters, time_steps_out)
        """
        activations = []
        for i, kernel in enumerate(self.filters):
            # 2D Convolution (Correlation)
            # ...existing code...
            if kernel.shape[0] != cochleogram.shape[0]:
                 # If kernel is not full height, we pad it or warn.
                 # For this simulation, we assume STRFGenerator makes full height kernels.
                 pass

            response = signal.correlate2d(cochleogram, kernel, mode='valid')
            # response shape: (input_h - kernel_h + 1, input_w - kernel_w + 1)
            # If kernel_h == input_h, shape is (1, time_steps_out)
            
            activation_signal = response.flatten()
            
            if self.use_spiking and i < len(self.neurons):
                # Convert analog activation to spikes
                # Scale activation to be reasonable current
                # Assuming activation is roughly 0-10 range, threshold is 1.0
                # We might need gain control here too
                
                # Rectify activation (neurons don't fire on negative correlation usually, or we assume baseline rate)
                # For this simple model, we rectify.
                rectified_activation = np.maximum(0, activation_signal)
                
                current = rectified_activation * 0.5 
                spikes = self.neurons[i].process_signal(current)
                activations.append(spikes)
            else:
                activations.append(activation_signal)
            
        return np.array(activations)

    def process_dilated(self, cochleogram: np.ndarray, dilation_rate: int = 2) -> np.ndarray:
        """
        Processes input with dilated convolutions to capture longer temporal context.
        Addresses the "Stationarity Problem" and "Harmonic Trap".
        
        Args:
            cochleogram: Shape (channels, time_steps)
            dilation_rate: Dilation factor for temporal dimension.
            
        Returns:
            activations: Shape (num_filters, time_steps_out)
        """
        activations = []
        for kernel in self.filters:
            # Simulate dilation by inserting zeros in the kernel (temporal axis only)
            # kernel shape: (channels, time)
            k_channels, k_time = kernel.shape
            dilated_width = k_time + (k_time - 1) * (dilation_rate - 1)
            dilated_kernel = np.zeros((k_channels, dilated_width))
            
            # Fill dilated kernel
            dilated_kernel[:, ::dilation_rate] = kernel
            
            # Convolve
            response = signal.correlate2d(cochleogram, dilated_kernel, mode='valid')
            activations.append(response.flatten())
            
        return np.array(activations)

class AuditoryCortex:
    """
    Manages the tonotopic map of expert modules.
    Implements Phase 2.1 (Tonotopy).
    """
    def __init__(self, num_modules: int = 10, num_channels: int = 64):
        self.modules = []
        self.num_channels = num_channels
        
        # Create tonotopic modules
        freq_step = max(1, self.num_channels // num_modules)
        for i in range(num_modules):
            start = i * freq_step
            end = min(start + freq_step, self.num_channels)
            if start >= self.num_channels:
                break
            module = AuditoryExpertModule(f"A1_Module_{i}", (start, end))
            self.modules.append(module)
            
    def initialize_filters(self, generator: STRFGenerator):
        """
        Initializes filters for all modules based on their tonotopic position.
        """
        for module in self.modules:
            # Center frequency for this module
            bf_idx = (module.best_freq_range[0] + module.best_freq_range[1]) // 2
            
            # Generate a few diverse STRFs for this BF
            # 1. Up-sweep (FM)
            strf1 = generator.generate_gabor_strf(bf_idx, 5.0, 10.0, 0.1, 1)
            module.add_filter(strf1)
            
            # 2. Down-sweep (FM)
            strf2 = generator.generate_gabor_strf(bf_idx, 5.0, 10.0, 0.1, -1)
            module.add_filter(strf2)
            
            # 3. Static tone (Pure Tone)
            strf3 = generator.generate_gabor_strf(bf_idx, 3.0, 0.0, 0.0, 1)
            module.add_filter(strf3)
            
            # 4. Broad band click (Temporal)
            strf4 = generator.generate_gabor_strf(bf_idx, 20.0, 20.0, 0.0, 1)
            module.add_filter(strf4)

    def process_input(self, cochleogram: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Passes input through all modules.
        """
        results = {}
        for module in self.modules:
            results[module.module_id] = module.process(cochleogram)
        return results

class AuditoryReadout:
    """
    Maps high-dimensional cortical activity to abstract concepts.
    Implements Phase 5.1 (Perception-to-Concept Interface).
    """
    def __init__(self, cortex: AuditoryCortex):
        self.cortex = cortex
        
    def decode_emotion(self, activations: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Heuristic decoding of emotional valence/arousal based on spectral/temporal energy.
        
        Hypothesis:
        - High temporal modulation (>10Hz) + High Frequency -> High Arousal (Fear/Anger)
        - Low temporal modulation (<4Hz) + Low Frequency -> Low Arousal (Sadness/Calm)
        """
        total_energy = 0.0
        high_freq_energy = 0.0
        fast_mod_energy = 0.0
        
        num_modules = len(self.cortex.modules)
        
        for i, (mod_id, act) in enumerate(activations.items()):
            # act shape: (num_filters, time)
            energy = np.mean(np.abs(act))
            total_energy += energy
            
            # High Frequency check (upper half of modules)
            if i > num_modules / 2:
                high_freq_energy += energy
                
            # Fast modulation check
            # We assume filters are ordered or we check their properties.
            # Since we don't have metadata attached to filter outputs easily here,
            # we'll use a simplified heuristic: Variance over time often correlates with modulation.
            # Or we rely on the fact that we initialized specific filters.
            # For this "Pflasterprinzip", we'll use the raw energy as a proxy for "activity".
            
        if total_energy == 0:
            return {"valence": 0.0, "arousal": 0.0}
            
        # Normalize
        norm_high_freq = high_freq_energy / total_energy
        
        # Arousal map
        arousal = 2.0 * (norm_high_freq - 0.5) # -1 to 1
        
        # Valence is hard without semantic training. 
        # We'll default to neutral unless extreme arousal.
        valence = 0.0
        if arousal > 0.8:
            valence = -0.5 # High arousal often negative (alarm)
            
        return {
            "valence": np.clip(valence, -1.0, 1.0),
            "arousal": np.clip(arousal, -1.0, 1.0),
            "intensity": np.clip(total_energy / 1000.0, 0.0, 1.0) # Arbitrary scaling
        }

class LIFNeuron:
    """
    Leaky Integrate-and-Fire Neuron model.
    Implements Phase 3.3 (Spiking Dynamics).
    """
    def __init__(self, tau: float = 20.0, threshold: float = 1.0, v_reset: float = 0.0, dt: float = 1.0):
        self.tau = tau # Membrane time constant (ms)
        self.threshold = threshold
        self.v_reset = v_reset
        self.dt = dt # Simulation step (ms)
        self.v = v_reset # Membrane potential
        
    def update(self, input_current: float) -> int:
        """
        Updates neuron state.
        Returns 1 if spike, 0 otherwise.
        """
        # dV/dt = -(V - V_rest)/tau + I
        # Discrete update: V(t+1) = V(t) + dt/tau * (-(V(t) - V_rest) + Input)
        # Assuming V_rest = 0
        
        dv = (self.dt / self.tau) * (-self.v + input_current)
        self.v += dv
        
        spike = 0
        if self.v >= self.threshold:
            spike = 1
            self.v = self.v_reset
            
        return spike

    def process_signal(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Processes a whole signal array and returns spike train.
        """
        spikes = []
        for current in input_signal:
            spikes.append(self.update(current))
        return np.array(spikes)
