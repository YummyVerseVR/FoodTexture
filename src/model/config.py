import os

# Augmentation configuration
AUGMENT_DATA_TARGET_DIR = "./audio/"
AUGMENT_DATA_OUTPUT_DIR = "./augmented_audio/"

# Preprocess configuration
RAW_DATA_DIR = "audio/"
PROCESSED_DATA_DIR = "dataset/"

# Paths for saving models and data
GENRATED_DATA_DIR = "generated/"
GENERATED_AUDIO_DIR = os.path.join(GENRATED_DATA_DIR, "audio/")
GENERATED_IMAGE_DIR = os.path.join(GENRATED_DATA_DIR, "image/")
PROCESSED_DATA_DIR = "augmented_dataset/"
WORD2VEC_MODEL_DIR = "models/word2vec"
VOCODER_MODEL_DIR = "models/vocoder"
CHECKPOINT_DIR = "models/checkpoints/"

WORD2VEC_MODEL_URL = (
    "http://nathaniel.iruma.arc/modelfile/GoogleNews-vectors-negative300.bin"
)

GENERATOR_MODEL_PATH = "./models/generator/latest.pth"
LATEST_CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/latest.pth"
WORD2VEC_MODEL_PATH = os.path.join(
    WORD2VEC_MODEL_DIR, "GoogleNews-vectors-negative300.bin"
)
VOCODER_MODEL_DATA_PATH = os.path.join(
    VOCODER_MODEL_DIR, "ljspeech_parallel_wavegan.v1.tar.gz"
)
VOCODER_MODEL_PATH = os.path.join(
    VOCODER_MODEL_DIR, "ljspeech_parallel_wavegan.v1/checkpoint-400000steps.pkl"
)

AUGMENTATIONS_PER_FILE = 3
NUM_IMAGES_PER_WORD = 1

# Hyperparameters for cGAN (config.py)
W2V_DIM = 300  # Dimension of Word2Vec vectors
LATENT_DIM = 100  # Dimension of the noise vector

# List of words to generate images and audio for
INPUT_WORDS = [
    "cookie",
    "apple",
    "chips",
    "water",
    "cracker",
    "lettuce",
    "bacon",
    "aloe",
]

# cGAN configuration
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
SPECTROGRAM_FIXED_LENGTH = 128
