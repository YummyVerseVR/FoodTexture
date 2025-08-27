import torch
from gensim.models import KeyedVectors

from model.cGAN import Generator
from model.config import LATENT_DIM, WORD2VEC_MODEL_PATH, GENERATOR_MODEL_PATH, W2V_DIM


# class Word2Vec:
#     def __init__(self, model_path: str = WORD2VEC_MODEL_PATH):
#         self.__model = KeyedVectors.load_word2vec_format(model_path, binary=True)

#     def get_vector(self, word: str) -> torch.FloatTensor:
#         if word in self.__model:
#             return torch.FloatTensor(self.__model[word].copy())
#         else:
#             print(f"WARNING: '{word}' not in Word2Vec vocabulary.")
#             return torch.zeros(self.__model.vector_size)


class Word2Vec:
    def __init__(self, model_path: str = WORD2VEC_MODEL_PATH):
        self.vector_size = W2V_DIM

    def get_vector(self, word: str) -> torch.FloatTensor:
        return torch.randn(self.vector_size)


# class SpectrogramGenerator:
#     def __init__(
#         self,
#         model_path: str = GENERATOR_MODEL_PATH,
#         device: torch.device = torch.device("cuda"),
#     ):
#         self.__device = device
#         self.__generator = Generator().to(self.__device)
#         self.__generator.load_state_dict(
#             torch.load(model_path, map_location=self.__device)
#         )
#         self.__generator.eval()

#     def generate(self, word_vector: torch.FloatTensor) -> torch.FloatTensor:
#         with torch.no_grad():
#             noise = torch.randn(1, LATENT_DIM).to(self.__device)
#             word_vector = word_vector.to(self.__device).unsqueeze(0)
#             fake_spec = self.__generator(noise, word_vector)
#             return (fake_spec + 1) / 2.0


class SpectrogramGenerator:
    def __init__(
        self,
        model_path: str = GENERATOR_MODEL_PATH,
        device: torch.device = torch.device("cuda"),
    ):
        self.__device = device
        self.spec_shape = (1, 80, 400)

    def generate(self, word_vector: torch.FloatTensor) -> torch.FloatTensor:
        return torch.rand(self.spec_shape)


class Vocoder:
    def __init__(self):
        # self.__vocoder = torch.load(model_path, map_location=self.__device)
        ...

    def generate(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        # with torch.no_grad():
        #     waveform = self.__vocoder.spectrogram_to_waveform(spectrogram)
        #     return waveform
        return torch.randn(22050 * 3)
