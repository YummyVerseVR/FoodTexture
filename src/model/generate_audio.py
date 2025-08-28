import torch
import soundfile as sf
from PIL import Image
from parallel_wavegan.utils import load_model
import torchvision.transforms as T

from config import (
    VOCODER_MODEL_PATH,
    GENERATED_AUDIO_DIR,
    GENERATED_IMAGE_DIR,
    INPUT_WORDS,
)


def generate():
    # 1. Load pre-trained vocoder model
    vocoder = load_model(VOCODER_MODEL_PATH).to("cuda")
    vocoder.eval()

    if torch.cuda.is_available():
        vocoder = vocoder.cuda()

    for word in INPUT_WORDS:
        # 2. Path to mel-spectrogram image
        image_path = f"{GENERATED_IMAGE_DIR}/{word}.png"
        img = Image.open(image_path).convert("L")  # Load as grayscale

        # Size: (n_mels=80, time_steps)
        to_tensor = T.Compose(
            [
                T.Resize((80, img.height)),  # Resize to (80, T)
                T.ToTensor(),
            ]
        )
        mel_tensor = to_tensor(img).squeeze(0)  # [80, T]

        # 3. Convert [0,1] to [-80,0] dB
        mel_db = mel_tensor * 80.0 - 80.0

        mel_db = mel_db.T  # TODO: Check why this line is needed!!

        mel_db = mel_db.float()  # Ensure float32 type
        mel_db = mel_db.to(next(vocoder.parameters()).device)

        # 4. Generate waveform
        with torch.no_grad():
            if torch.cuda.is_available():
                mel_db = mel_db.cuda()
            waveform = vocoder.inference(mel_db)  # [1, T]

        # 5. Save waveform
        waveform = waveform.squeeze(0).cpu().numpy()
        sf.write(f"{GENERATED_AUDIO_DIR}/{word}.wav", waveform, 22050)
        print(f"Saved {word}.wav")


if __name__ == "__main__":
    generate()
