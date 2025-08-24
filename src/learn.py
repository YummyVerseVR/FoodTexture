import os
import random
import numpy
import torch
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from cGAN import PreprocessedFoodSoundDataset, Generator, Discriminator, LATENT_DIM

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda")
    print(f"Using device: {device}")

    PROCESSED_DATA_DIR = "dataset/"
    # Use the modified Dataset class
    dataset = PreprocessedFoodSoundDataset(PROCESSED_DATA_DIR)
    # Increase num_workers for faster data loading
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=0.000001, betas=(0.5, 0.999)
    )
    # --- 4. Load Checkpoint if it exists ---
    start_epoch = 0
    CHECKPOINT_PATH = "checkpoint.pth"

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found at {CHECKPOINT_PATH}. Resuming training.")
        checkpoint = torch.load(CHECKPOINT_PATH)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    num_epochs = 100

    for epoch in range(num_epochs):
        for i, (real_specs, word_vecs) in enumerate(dataloader):
            real_specs = real_specs.to(device)
            word_vecs = word_vecs.to(device)
            batch_size = real_specs.size(0)

            # lebels: true=1, fake=0
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # learn discriminator
            optimizer_d.zero_grad()

            # 1. Learn with real spectrograms
            outputs = discriminator(real_specs.unsqueeze(1), word_vecs)
            d_loss_real = criterion(outputs, real_labels)

            # 2. Learn with fake spectrograms
            noise = torch.randn(batch_size, LATENT_DIM).to(device)
            fake_specs = generator(noise, word_vecs)
            outputs = discriminator(
                fake_specs.detach(), word_vecs
            )  # detach to avoid training G on these labels
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # learn generator
            optimizer_g.zero_grad()

            outputs = discriminator(fake_specs, word_vecs)
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_g.step()

            if (i + 1) % 50 == 0:
                print(
                    f"  Batch [{i + 1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
                )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
        )

        # --- 6. Save Checkpoint periodically ---
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            print(f"Saving checkpoint at epoch {epoch + 1}...")
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "g_loss": g_loss,
                    "d_loss": d_loss,
                },
                CHECKPOINT_PATH,
            )

    # Save models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Models saved.")
