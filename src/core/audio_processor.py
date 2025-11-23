import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from typing import Tuple, Optional, Union
import os
from core.logger import logger

class AudioLoader:
    """
    Handles loading and preprocessing of audio files for the auditory cortex simulation.
    Implements Phase 1.1 of the migration plan.
    """
    def __init__(self, target_sr: int = 16000, target_rms: float = 0.1, pre_emphasis: float = 0.97):
        self.target_sr = target_sr
        self.target_rms = target_rms
        self.pre_emphasis = pre_emphasis

    def apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies a first-order pre-emphasis filter to flatten spectral tilt.
        y[n] = x[n] - alpha * x[n-1]
        """
        if self.pre_emphasis <= 0:
            return audio
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads an audio file (WAV) and resamples it to target_sr.
        
        Args:
            file_path: Path to the audio file.
            
        Returns:
            audio (np.ndarray): Normalized audio signal (float32).
            sr (int): Sample rate.
        """
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Load using scipy
            sr, audio = wav.read(file_path)
            
            # Convert to float32 and normalize to [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128.0) / 128.0
            elif audio.dtype == np.float32:
                pass # Already float32
            else:
                audio = audio.astype(np.float32) # Fallback
                
            # Handle stereo to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            # Resample if necessary
            if sr != self.target_sr:
                num_samples = int(len(audio) * self.target_sr / sr)
                audio = signal.resample(audio, num_samples)
                sr = self.target_sr
            
            # Apply Pre-Emphasis (Phase 1.4 - Engineer's Fix)
            audio = self.apply_pre_emphasis(audio)
                
            # RMS Normalization (Phase 1.1)
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                audio = audio * (self.target_rms / rms)
                
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise e

class AuditoryFeatures:
    """
    Extracts biologically plausible auditory features (Cochleograms).
    Implements Phase 1.2 and 1.3 of the migration plan.
    """
    def __init__(self, sr: int = 16000, num_channels: int = 64, min_freq: float = 100, max_freq: float = 8000):
        self.sr = sr
        self.num_channels = num_channels
        self.min_freq = min_freq
        self.max_freq = max_freq
        
    def _erb_space(self, low_freq: float, high_freq: float, num_channels: int) -> np.ndarray:
        """
        Computes center frequencies equally spaced on the ERB scale.
        """
        ear_q = 9.26449 # Glasberg and Moore Parameters
        min_bw = 24.7
        
        # Convert to ERB scale
        erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
        erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
        
        # Equal spacing
        erb_centers = np.linspace(erb_low, erb_high, num_channels)
        
        # Convert back to Hz
        center_freqs = 1000 * ((10**(erb_centers / 21.4)) - 1) / 4.37
        return center_freqs

    def gammatone_filterbank(self, audio: np.ndarray) -> np.ndarray:
        """
        Applies a Gammatone filterbank to the audio.
        Returns:
            filtered_audio (np.ndarray): Shape (num_channels, time_steps)
        """
        center_freqs = self._erb_space(self.min_freq, self.max_freq, self.num_channels)
        
        # Simple Gammatone Impulse Response implementation (approximate)
        # In a full implementation, we would use IIR filters for efficiency.
        # Here we use a simplified approach or rely on scipy.signal if available for design.
        # Since scipy.signal.gammatone is new, we'll use a standard bandpass filterbank 
        # distributed on ERB scale as a robust proxy for this "Pflasterprinzip" implementation.
        
        filtered_signals = []
        for freq in center_freqs:
            # Bandwidth approximation
            erb = 24.7 * (4.37 * freq / 1000 + 1)
            
            # Clamp frequencies to Nyquist
            nyquist = self.sr / 2.0
            low = max(0.1, freq - erb/2)
            high = min(nyquist - 1.0, freq + erb/2)
            
            if low >= high:
                # Skip invalid filters (should not happen with proper range)
                continue
                
            b, a = signal.iirfilter(N=4, Wn=[low, high], fs=self.sr, btype='bandpass')
            filtered = signal.lfilter(b, a, audio)
            filtered_signals.append(filtered)
            
        return np.stack(filtered_signals)

    def pcen(self, magnitude: np.ndarray, alpha: float = 0.98, delta: float = 2.0, r: float = 0.5, eps: float = 1e-6) -> np.ndarray:
        """
        Per-Channel Energy Normalization (PCEN).
        Implements Phase 1.3 (Correction).
        
        M = (E / (eps + M_smooth)**alpha + delta)**r - delta**r
        """
        # Simple smoothing (IIR)
        m_smooth = np.zeros_like(magnitude)
        # Assuming magnitude is (channels, time)
        # We need to iterate over time for causal smoothing
        # Or use lfilter
        
        b = [1 - alpha]
        a = [1, -alpha]
        
        # Apply smoothing per channel
        m_smooth = signal.lfilter(b, a, magnitude, axis=1)
        
        # PCEN formula
        pcen_out = (magnitude / (eps + m_smooth)**alpha + delta)**r - delta**r
        return pcen_out

    def compute_cochleogram(self, audio: np.ndarray, hop_length_ms: int = 10) -> np.ndarray:
        """
        Generates a Cochleogram from audio.
        
        Steps:
        1. Gammatone Filterbank
        2. Hilbert Transform (Envelope)
        3. Downsampling
        4. PCEN
        """
        # 1. Filterbank
        filtered = self.gammatone_filterbank(audio)
        
        # 2. Envelope (Hilbert)
        analytic_signal = signal.hilbert(filtered, axis=1)
        envelope = np.abs(analytic_signal)
        
        # 3. Downsampling (Average pooling)
        hop_samples = int(self.sr * hop_length_ms / 1000)
        # Reshape to (channels, num_frames, hop_samples) and mean
        num_frames = envelope.shape[1] // hop_samples
        envelope_trimmed = envelope[:, :num_frames * hop_samples]
        envelope_reshaped = envelope_trimmed.reshape(self.num_channels, num_frames, hop_samples)
        downsampled = np.mean(envelope_reshaped, axis=2)
        
        # 4. PCEN
        cochleogram = self.pcen(downsampled)
        
        return cochleogram

    def invert_pcen(self, pcen_out: np.ndarray, alpha: float = 0.98, delta: float = 2.0, r: float = 0.5) -> np.ndarray:
        """
        Approximate inversion of PCEN to recover magnitude envelope.
        Note: Exact inversion requires the smoothing history, which is lost.
        We assume steady state for approximation: M_smooth approx M.
        
        (M / (M**alpha) + delta)**r - delta**r = PCEN
        Let Y = PCEN + delta**r
        Y**(1/r) = M / M**alpha + delta
        Y**(1/r) - delta = M**(1-alpha)
        M = (Y**(1/r) - delta)**(1/(1-alpha))
        """
        y = pcen_out + delta**r
        # Avoid negative bases
        y = np.maximum(y, 0)
        
        term = y**(1/r) - delta
        term = np.maximum(term, 1e-6) # Avoid zeros
        
        magnitude = term**(1/(1-alpha))
        return magnitude

    def reconstruct_from_cochleogram(self, cochleogram: np.ndarray, hop_length_ms: int = 10, iterations: int = 10) -> np.ndarray:
        """
        Reconstructs audio from a cochleogram using sinusoidal modeling.
        Useful for 'Sonification' debugging.
        
        Args:
            cochleogram: Shape (channels, frames)
            hop_length_ms: Hop length used in generation.
            
        Returns:
            audio: Reconstructed time-domain signal.
        """
        num_channels, num_frames = cochleogram.shape
        hop_samples = int(self.sr * hop_length_ms / 1000)
        output_len = num_frames * hop_samples
        
        # 1. Invert PCEN (Approximate)
        envelope = self.invert_pcen(cochleogram)
        
        # 2. Upsample envelopes
        # Create time indices for interpolation
        t_frames = np.arange(num_frames) * hop_samples
        t_samples = np.arange(output_len)
        
        envelopes_upsampled = np.zeros((num_channels, output_len))
        for i in range(num_channels):
            envelopes_upsampled[i] = np.interp(t_samples, t_frames, envelope[i])
            
        # 3. Modulate Sine Waves
        center_freqs = self._erb_space(self.min_freq, self.max_freq, self.num_channels)
        reconstructed = np.zeros(output_len)
        
        time_axis = np.arange(output_len) / self.sr
        
        for i, freq in enumerate(center_freqs):
            # Generate carrier
            carrier = np.sin(2 * np.pi * freq * time_axis)
            # Modulate
            reconstructed += carrier * envelopes_upsampled[i]
            
        # Normalize
        if np.max(np.abs(reconstructed)) > 0:
            reconstructed /= np.max(np.abs(reconstructed))
            
        return reconstructed
