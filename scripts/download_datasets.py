import os
import shutil
import sys
from pathlib import Path
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm
import io

# Add src to path to use logger if needed, but standalone script is fine too
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core.logger import logger

# Emotion mapping for EMO-DB
# Filename format: 03a03Wb.wav -> Position 6 (index 5) is emotion
EMODB_MAPPING = {
    'W': 'anger',       # Wut
    'L': 'boredom',     # Langeweile
    'E': 'disgust',     # Ekel
    'A': 'fear',        # Angst
    'F': 'happiness',   # Freude
    'T': 'sadness',     # Trauer
    'N': 'neutral'      # Neutral
}

def download_emodb(output_dir: str):
    """
    Downloads the EMO-DB dataset from Hugging Face and organizes it by emotion.
    """
    print(f"Downloading EMO-DB to {output_dir}...")
    
    try:
        # Load dataset from Hugging Face
        # renumics/emodb contains the audio and metadata
        dataset = load_dataset("renumics/emodb", split="train")
        # Disable automatic decoding to avoid torchcodec dependency issues
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face: {e}")
        print("Error: Could not load dataset. Make sure 'datasets' and 'soundfile' are installed.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    
    print("Processing and saving files...")
    for item in tqdm(dataset):
        # With decode=False, item['audio'] contains {'bytes': b'...', 'path': '...'}
        audio_info = item['audio']
        audio_bytes = audio_info.get('bytes')
        
        # If we have raw bytes, we can write them directly if they are a valid file format (WAV)
        # Or we can decode them using soundfile/librosa if needed.
        # EMO-DB files are WAV.
        
        # Get emotion label
        emotion_label = item.get('emotion')
        
        # If emotion is an integer index, we need the class label
        if isinstance(emotion_label, int):
            emotion_name = dataset.features['emotion'].int2str(emotion_label)
        else:
            emotion_name = str(emotion_label)

        # Normalize emotion name to English lowercase
        target_folder = output_path / emotion_name
        target_folder.mkdir(exist_ok=True)
        
        # Generate a filename
        filename = Path(audio_info['path']).name if 'path' in audio_info else f"emodb_{count}.wav"
        target_file = target_folder / filename
        
        # Save audio
        if audio_bytes:
            with open(target_file, "wb") as f:
                f.write(audio_bytes)
        else:
            # Fallback if bytes are missing (should not happen with decode=False)
            print(f"Warning: No bytes found for {filename}")
            continue
            
        count += 1

    print(f"Successfully saved {count} files to {output_dir}")

if __name__ == "__main__":
    # Default path: data/emodb
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
    emodb_dir = os.path.join(base_dir, 'emodb')
    
    download_emodb(emodb_dir)
