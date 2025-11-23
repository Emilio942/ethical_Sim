import os
import sys
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.audio_augmentor import AudioAugmentor
from core.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Synthesize new audio data from existing samples.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source WAV files (Gold Data)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save synthesized files")
    parser.add_argument("--variations", type=int, default=5, help="Number of variations to generate per file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    augmentor = AudioAugmentor(target_sr=16000)
    
    # Find all wav files
    wav_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
                
    print(f"Found {len(wav_files)} source files.")
    print(f"Generating {args.variations} variations per file...")
    
    count = 0
    for file_path in tqdm(wav_files, desc="Synthesizing"):
        # Load
        audio = augmentor.load_audio(file_path)
        if audio is None:
            continue
            
        # Generate
        variations = augmentor.generate_variations(audio, num_variations=args.variations)
        
        # Save
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Preserve subdirectory structure if needed, but for now flat output or mirror
        # Let's mirror the folder structure relative to input_dir
        rel_path = os.path.relpath(os.path.dirname(file_path), args.input_dir)
        save_dir = os.path.join(args.output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)
        
        for i, var_audio in enumerate(variations):
            out_name = f"{base_name}_synth_{i+1}.wav"
            out_path = os.path.join(save_dir, out_name)
            augmentor.save_audio(var_audio, out_path)
            count += 1
            
    print(f"Done. Generated {count} new synthetic samples in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
