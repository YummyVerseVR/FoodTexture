import torch
from gensim.models import KeyedVectors
from parallel_wavegan.utils import load_model

from model.cGAN import Generator
from model.config import (
    LATENT_DIM,
    WORD2VEC_MODEL_PATH,
    GENERATOR_MODEL_PATH,
    VOCODER_MODEL_PATH,
)


class Word2Vec:
    def __init__(self, model_path: str = WORD2VEC_MODEL_PATH):
        self.__model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def get_vector(self, word: str) -> torch.FloatTensor:
        if word in self.__model:
            return torch.FloatTensor(self.__model[word].copy())
        else:
            print(f"WARNING: '{word}' not in Word2Vec vocabulary.")
            return torch.zeros(self.__model.vector_size)


class SpectrogramGenerator:
    def __init__(
        self,
        model_path: str = GENERATOR_MODEL_PATH,
        device: torch.device = torch.device("cuda"),
    ):
        self.__device = device
        self.__generator = Generator().to(self.__device)
        self.__generator.load_state_dict(
            torch.load(model_path, map_location=self.__device)
        )
        self.__generator.eval()

    def generate(self, word_vector: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            noise = torch.randn(1, LATENT_DIM).to(self.__device)
            word_vector = word_vector.to(self.__device).unsqueeze(0)
            fake_spec = self.__generator(noise, word_vector)
            return (fake_spec + 1) / 2.0


class Vocoder:
    def __init__(
        self,
        model_path: str = VOCODER_MODEL_PATH,
        device: torch.device = torch.device("cuda"),
    ):
        self.__device = device
        self.__vocoder = load_model(model_path).to(device)
        self.__vocoder.eval()

        if torch.cuda.is_available():
            self.__vocoder = self.__vocoder.cuda()

    def generate(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            if torch.cuda.is_available():
                spectrogram = spectrogram.cuda()
            waveform = self.__vocoder.inference(spectrogram)
            return waveform.squeeze(0).cpu()
