# preprocess.py

import torch
import librosa
import numpy as np
from gensim.models import KeyedVectors
import os
from tqdm import tqdm  # Import tqdm for progress bars

from config import (
    WORD2VEC_MODEL_PATH,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    SPECTROGRAM_FIXED_LENGTH,
)


def preprocess_data():
    """
    Converts audio data and word labels into pairs of
    mel spectrograms and Word2Vec vectors, and saves them as .pt files.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print("Loading Word2Vec model...")
    w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
    print("Model loaded.")

    file_counter = 0

    # Use tqdm to display progress
    for label in tqdm(os.listdir(RAW_DATA_DIR), desc="Processing Labels"):
        label_dir = os.path.join(RAW_DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        # Check if the word exists in the Word2Vec vocabulary
        if label not in w2v_model:
            print(f"WARNING: '{label}' not in Word2Vec vocabulary. Skipping.")
            continue

        # Get the Word2Vec vector only once per word
        word_vector = torch.FloatTensor(w2v_model[label].copy())

        for filename in os.listdir(label_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(label_dir, filename)

                # 1. Load audio file and convert to mel spectrogram
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                mel_spec = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # 2. Fix the length (padding or trimming)
                if mel_spec_db.shape[1] < SPECTROGRAM_FIXED_LENGTH:
                    pad_width = SPECTROGRAM_FIXED_LENGTH - mel_spec_db.shape[1]
                    mel_spec_db = np.pad(
                        mel_spec_db,
                        ((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=-80.0,
                    )
                else:
                    mel_spec_db = mel_spec_db[:, :SPECTROGRAM_FIXED_LENGTH]

                spectrogram_tensor = torch.FloatTensor(mel_spec_db.copy())

                # 3. Save the spectrogram and word vector pair as a dictionary
                data_pair = {
                    "spectrogram": spectrogram_tensor,
                    "word_vector": word_vector,
                }

                save_path = os.path.join(PROCESSED_DATA_DIR, f"{file_counter}.pt")
                torch.save(data_pair, save_path)
                file_counter += 1

    print(
        f"\nPreprocessing complete. Saved {file_counter} files to '{PROCESSED_DATA_DIR}'."
    )


if __name__ == "__main__":
    preprocess_data()
