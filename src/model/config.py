# Data augmentation configuration (augment_data.py)
INPUT_DIR = "./audio/"
OUTPUT_DIR = "./augmented_audio/"
AUGMENTATIONS_PER_FILE = 3

# Hyperparameters for cGAN (config.py)
W2V_DIM = 300  # Dimension of Word2Vec vectors
LATENT_DIM = 100  # Dimension of the noise vector

# Paths and settings for generator (generator.py)
GENERATOR_MODEL_PATH = "./models/generator/latest.pth"  # CHANGE THIS to your model file
# Directory to save the output images
OUTPUT_DIR = "./generated_images"
# List of words to generate images for
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
# Number of images to generate per word
NUM_IMAGES_PER_WORD = 1

# Learing configuration (learn.py)
PROCESSED_DATA_DIR = "augmented_dataset/"
CHECKPOINT_FOLDER = "models/checkpoints/"
LATEST = f"{CHECKPOINT_FOLDER}/latest.pth"

# Preprocessing configuration (preprocess.py)
# Path to the pre-trained Word2Vec model
WORD2VEC_MODEL_PATH = "w2v/GoogleNews-vectors-negative300.bin"
# Directory of the raw audio dataset
RAW_DATA_DIR = "audio/"
# Directory to save the preprocessed data
PROCESSED_DATA_DIR = "dataset/"
# Audio processing hyperparameters (must match the training script)
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
SPECTROGRAM_FIXED_LENGTH = 128
