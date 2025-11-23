import numpy as np
import librosa
import soundfile as sf
import os
from typing import List, Optional
from core.logger import logger

class AudioAugmentor:
    """
    A tool to synthesize new audio data from existing 'gold' samples.
    Applies various augmentations to increase dataset size and diversity.
    """
    
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load_audio(self, file_path: str) -> np.ndarray:
        try:
            audio, _ = librosa.load(file_path, sr=self.target_sr)
            return audio
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None

    def save_audio(self, audio: np.ndarray, output_path: str):
        try:
            sf.write(output_path, audio, self.target_sr)
        except Exception as e:
            logger.error(f"Failed to save {output_path}: {e}")

    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Adds white noise to the audio."""
        noise = np.random.randn(len(audio))
        augmented_data = audio + noise_factor * noise
        # Cast back to same type if needed, but float is fine for librosa
        return augmented_data

    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """
        Changes speed without changing pitch.
        rate > 1.0: faster
        rate < 1.0: slower
        """
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """
        Changes pitch without changing speed.
        n_steps: fractional steps (semitones)
        """
        return librosa.effects.pitch_shift(audio, sr=self.target_sr, n_steps=n_steps)

    def change_gain(self, audio: np.ndarray, gain_factor: float = 1.0) -> np.ndarray:
        """Changes volume."""
        return audio * gain_factor

    def shift_time(self, audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """
        Shifts audio in time (rolling). 
        shift_max: max shift in seconds
        """
        shift_samples = int(shift_max * self.target_sr)
        shift = np.random.randint(-shift_samples, shift_samples)
        return np.roll(audio, shift)

    def generate_variations(self, audio: np.ndarray, num_variations: int = 5) -> List[np.ndarray]:
        """
        Generates multiple variations of a single audio track using random augmentations.
        """
        variations = []
        
        for _ in range(num_variations):
            y = audio.copy()
            
            # Randomly apply augmentations
            if np.random.random() < 0.5:
                # Pitch: +/- 2 semitones
                steps = np.random.uniform(-2.0, 2.0)
                y = self.pitch_shift(y, n_steps=steps)
                
            if np.random.random() < 0.5:
                # Speed: 0.8x to 1.2x
                rate = np.random.uniform(0.8, 1.2)
                y = self.time_stretch(y, rate=rate)
                
            if np.random.random() < 0.5:
                # Noise
                y = self.add_noise(y, noise_factor=np.random.uniform(0.001, 0.01))
                
            if np.random.random() < 0.5:
                # Gain
                y = self.change_gain(y, gain_factor=np.random.uniform(0.7, 1.3))
                
            variations.append(y)
            
        return variations
