import subprocess
import os
from parallel_wavegan.utils import download_pretrained_model

from config import (
    WORD2VEC_MODEL_DIR,
    VOCODER_MODEL_DIR,
    WORD2VEC_MODEL_URL,
    WORD2VEC_MODEL_PATH,
    VOCODER_MODEL_DATA_PATH,
)


def prepare_directories():
    """Create necessary directories for models."""
    print("Preparing directories...")
    os.makedirs("models", exist_ok=True)
    os.makedirs(WORD2VEC_MODEL_DIR, exist_ok=True)
    os.makedirs(VOCODER_MODEL_DIR, exist_ok=True)


def check_model_exists(path):
    """Check if a model file exists."""
    return os.path.isfile(path)


def download_word2vec_model():
    """Download the Google News word2vec model."""
    print("Downloading Word2Vec model...")
    subprocess.run(
        [
            "wget",
            WORD2VEC_MODEL_URL,
            "-P",
            WORD2VEC_MODEL_DIR,
        ]
    )


def download_vocoder_model():
    """Download the pretrained vocoder model."""
    print("Downloading Vocoder model...")
    download_pretrained_model("ljspeech_parallel_wavegan.v1", VOCODER_MODEL_DIR)


if __name__ == "__main__":
    prepare_directories()

    if not check_model_exists(WORD2VEC_MODEL_PATH):
        print("Word2Vec model not found.")
        download_word2vec_model()
    else:
        print("Word2Vec model already exists.")
        print("Skipping download.")

    if not check_model_exists(VOCODER_MODEL_DATA_PATH):
        print("Vocoder model not found.")
        download_vocoder_model()
    else:
        print("Vocoder model already exists.")
        print("Skipping download.")

    print("All models are ready.")
