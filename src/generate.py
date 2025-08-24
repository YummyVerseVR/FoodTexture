# generate.py

import torch
from torchvision.utils import save_image
from gensim.models import KeyedVectors
import os

# Import model and configuration from your other files
from cGAN import Generator
from preprocess import WORD2VEC_MODEL_PATH
from learn import LATENT_DIM

# --- Configuration ---
# Path to your saved Generator model file
GENERATOR_MODEL_PATH = "./generator.pth"  # CHANGE THIS to your model file
# Directory to save the output images
OUTPUT_DIR = "./generated_images"
# List of words to generate images for
INPUT_WORDS = ["cookie", "apple", "chips", "water", "cracker", "lettuce", "bacon"]
# Number of images to generate per word
NUM_IMAGES_PER_WORD = 4


def generate():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Setup Device ---
    device = torch.device("cuda")
    print(f"Using device: {device}")

    # --- 2. Load Word2Vec Model ---
    print("Loading Word2Vec model...")
    try:
        w2v_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
        print("Word2Vec model loaded.")
    except FileNotFoundError:
        print(f"ERROR: Word2Vec model not found at {WORD2VEC_MODEL_PATH}")
        return

    # --- 3. Load Trained Generator ---
    # Instantiate the model and move it to the device
    generator = Generator().to(device)

    print(f"Loading trained generator from {GENERATOR_MODEL_PATH}...")
    try:
        # Load the saved state dictionary
        generator.load_state_dict(torch.load(GENERATOR_MODEL_PATH, map_location=device))
        print("Generator loaded successfully.")
    except FileNotFoundError:
        print(
            "ERROR: Generator model not found. Please check the GENERATOR_MODEL_PATH."
        )
        return

    # Set the model to evaluation mode
    generator.eval()

    # --- 4. Generate and Save Images ---
    # Disable gradient calculations for inference
    with torch.no_grad():
        for word in INPUT_WORDS:
            print(f"Generating images for '{word}'...")

            # Check if word is in vocabulary
            if word not in w2v_model:
                print(f"  - WARNING: '{word}' not in Word2Vec vocabulary. Skipping.")
                continue

            # Prepare the input vectors for the batch
            word_vector = torch.FloatTensor(w2v_model[word].copy()).to(device)
            # Repeat the word vector for each image we want to generate
            word_vectors_batch = word_vector.repeat(NUM_IMAGES_PER_WORD, 1)

            # Create a batch of random noise vectors
            noise_batch = torch.randn(NUM_IMAGES_PER_WORD, LATENT_DIM).to(device)

            # Generate a batch of fake spectrograms
            fake_specs = generator(noise_batch, word_vectors_batch)

            # Normalize the output from [-1, 1] (Tanh range) to [0, 1] for image saving
            fake_specs = (fake_specs + 1) / 2.0

            # Save the batch of images in a grid
            save_path = os.path.join(OUTPUT_DIR, f"{word}.png")
            save_image(fake_specs, save_path, nrow=NUM_IMAGES_PER_WORD)

            print(f"  - Saved images to {save_path}")

    print("\nGeneration complete.")


if __name__ == "__main__":
    generate()
