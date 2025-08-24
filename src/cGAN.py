from preprocess import N_MELS, SPECTROGRAM_FIXED_LENGTH
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Hyperparameters
W2V_DIM = 300  # Dimension of Word2Vec vectors
LATENT_DIM = 100  # Dimension of the noise vector


class PreprocessedFoodSoundDataset(Dataset):
    def __init__(self, directory):
        """
        Points to the directory containing preprocessed .pt files.
        """
        self.file_paths = []
        for filename in os.listdir(directory):
            if filename.endswith(".pt"):
                self.file_paths.append(os.path.join(directory, filename))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Simply loads a .pt file and returns the data from the dictionary inside.
        """
        data_pair = torch.load(self.file_paths[idx])
        sp, v = data_pair["spectrogram"], data_pair["word_vector"]
        max, min = sp.max(), sp.min()
        norm = 2 * (sp - min) / (max - min) - 1
        return torch.Tensor(norm), v


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Input: LATENT_DIM (noise) + W2V_DIM (condition) -> [batch, 400, 1, 1]
            nn.ConvTranspose2d(
                LATENT_DIM + W2V_DIM, 512, kernel_size=(5, 4), stride=1, padding=0
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: [batch, 512, 5, 4]
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: [batch, 256, 10, 8]
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: [batch, 128, 20, 16]
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: [batch, 64, 40, 32]
            nn.ConvTranspose2d(
                64, 1, kernel_size=(4, 4), stride=(2, 4), padding=(1, 0)
            ),
            nn.Tanh(),
            # Output: [batch, 1, 80, 128]
        )

    def forward(self, noise, word_vec):
        # Concatenate noise and word vector
        combined = torch.cat((noise, word_vec), 1)
        # Reshape for convolutional layers: [batch, channels, height, width]
        combined = combined.unsqueeze(2).unsqueeze(3)
        return self.model(combined)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.spec_path = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.w2v_path = nn.Sequential(
            nn.Linear(W2V_DIM, 128 * (N_MELS // 4) * (SPECTROGRAM_FIXED_LENGTH // 4)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.final_path = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=(10, 16), stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, spec, word_vec):
        spec_features = self.spec_path(spec)
        w2v_features = self.w2v_path(word_vec)
        w2v_features = w2v_features.view(
            -1, 128, (N_MELS // 4), (SPECTROGRAM_FIXED_LENGTH // 4)
        )

        combined = torch.cat((spec_features, w2v_features), 1)
        return self.final_path(combined).view(-1, 1)
