# augment_data.py

import os
import soundfile as sf
from audiomentations import Compose, TimeStretch, Shift
from tqdm import tqdm
import numpy as np

from config import AUGMENT_TARGET_DIR, AUGMENT_DATA_OUTPUT_DIR, AUGMENTATIONS_PER_FILE


def augment_data():
    """
    Performs data augmentation on audio files in the INPUT_DIR.
    For each audio file, it generates multiple augmented versions using
    """
    print("Starting data augmentation...")
    print(f"input: {AUGMENT_TARGET_DIR}")
    print(f"output: {AUGMENT_DATA_OUTPUT_DIR}")
    print(f"num augment: {AUGMENTATIONS_PER_FILE}")

    # Define the augmentation pipeline
    augment_pipeline = Compose(
        [
            # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
        ]
    )

    for label in tqdm(os.listdir(AUGMENT_TARGET_DIR), desc="Processing Labels"):
        label_dir = os.path.join(AUGMENT_TARGET_DIR, label)

        if not os.path.isdir(label_dir):
            continue

        # Create output directory for the current label
        output_label_dir = os.path.join(AUGMENT_DATA_OUTPUT_DIR, label)
        os.makedirs(output_label_dir, exist_ok=True)

        # Loop through all .wav files in the current label directory
        for filename in os.listdir(label_dir):
            if filename.endswith(".wav"):
                input_file_path = os.path.join(label_dir, filename)

                # Load the original audio file
                try:
                    audio, sample_rate = sf.read(input_file_path)
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=1).astype(np.float32)
                except Exception as e:
                    print(f"Error: failed to load {input_file_path}: {e}")
                    continue

                # Save the augmented versions
                for i in range(AUGMENTATIONS_PER_FILE):
                    augmented_audio = augment_pipeline(
                        samples=audio, sample_rate=sample_rate
                    )

                    base_filename = os.path.splitext(filename)[0]
                    new_filename = f"{base_filename}_aug_{i + 1}.wav"
                    output_file_path = os.path.join(output_label_dir, new_filename)

                    sf.write(output_file_path, augmented_audio, sample_rate)

    print("Data augmentation completed.")


if __name__ == "__main__":
    augment_data()
