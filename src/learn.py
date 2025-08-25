import os
import random
import numpy
import torch
import torch.nn as nn
import datetime

from torch.utils.data import DataLoader
from cGAN import PreprocessedFoodSoundDataset, Generator, Discriminator, LATENT_DIM

import matplotlib.pyplot as plt


def setup():
    seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda")
    print(f"Using device: {device}")

    PROCESSED_DATA_DIR = "augmented_dataset/"
    # Use the modified Dataset class
    dataset = PreprocessedFoodSoundDataset(PROCESSED_DATA_DIR)
    # Increase num_workers for faster data loading
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)
    )

    return (
        device,
        dataloader,
        generator,
        discriminator,
        criterion,
        optimizer_g,
        optimizer_d,
    )


def save_checkpoint(epoch, checkpoint_data, checkpoint_folder):
    torch.save(checkpoint_data, f"{checkpoint_folder}/latest.pth")
    torch.save(checkpoint_data, f"{checkpoint_folder}/checkpoint-{epoch}.pth")


def cleanup_checkpoints(checkpoint_folder, max_checkpoints=5):
    checkpoints = [
        f
        for f in os.listdir(checkpoint_folder)
        if os.path.isfile(os.path.join(checkpoint_folder, f)) and f.endswith(".pth")
    ]
    checkpoints.remove("latest.pth")  # Keep the latest checkpoint

    if len(checkpoints) > max_checkpoints:
        checkpoints.sort(
            key=lambda x: os.path.getmtime(os.path.join(checkpoint_folder, x))
        )
        for ckpt in checkpoints[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_folder, ckpt))
            print(f"Removed old checkpoint: {ckpt}")


def cleanup_models(model_folder, max_models=5):
    models = [
        f
        for f in os.listdir(model_folder)
        if os.path.isfile(os.path.join(model_folder, f)) and f.endswith(".pth")
    ]
    models.remove("latest.pth")  # Keep the latest model

    if len(models) > max_models:
        models.sort(key=lambda x: os.path.getmtime(os.path.join(model_folder, x)))
        for mdl in models[:-max_models]:
            os.remove(os.path.join(model_folder, mdl))
            print(f"Removed old model: {mdl}")


def add_instance_noise(x, sigma=0.05):
    if sigma <= 0:
        return x
    n = torch.randn_like(x) * sigma
    return torch.clamp(x + n, -1.0, 1.0)


def run(
    device, dataloader, generator, discriminator, optimizer_d, optimizer_g, criterion
) -> None:
    # --- 4. Load Checkpoint if it exists ---
    start_epoch = 0
    CHECKPOINT_FOLDER = "models/checkpoints/"
    LATEST = f"{CHECKPOINT_FOLDER}/latest.pth"

    if os.path.exists(LATEST):
        print(f"Checkpoint found at {LATEST}. Resuming training.")
        checkpoint = torch.load(LATEST)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

        start_epoch = checkpoint["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    num_epochs = 2000
    # num_epochs = 1
    d_loss_list = []
    g_loss_list = []

    for epoch in range(num_epochs):
        try:
            dl = 0
            d_loss_sum = 0.0
            g_loss_sum = 0.0
            # for i, (real_specs, word_vecs) in tqdm(enumerate(dataloader), desc="Batch"):
            for i, (real_specs, word_vecs) in enumerate(dataloader):
                real_specs = real_specs.to(device)
                word_vecs = word_vecs.to(device)
                batch_size = real_specs.size(0)

                # lebels: true=1, fake=0
                real_labels = torch.ones(batch_size, 1).to(device) * 0.98
                fake_labels = torch.zeros(batch_size, 1).to(device)

                # learn discriminator
                optimizer_d.zero_grad()

                # 1. Learn with real spectrograms
                outputs = discriminator(
                    add_instance_noise(real_specs.unsqueeze(1)), word_vecs
                )
                d_loss_real = criterion(outputs, real_labels)

                # 2. Learn with fake spectrograms
                noise = torch.randn(batch_size, LATENT_DIM).to(device)
                fake_specs = generator(noise, word_vecs)
                # detach to avoid training G on these labels
                outputs = discriminator(
                    add_instance_noise(fake_specs.detach()), word_vecs
                )
                d_loss_fake = criterion(outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()

                # Adpative training for generator
                if d_loss.item() < 0.1:
                    optimizer_g.zero_grad()
                    # ここで新ノイズ＆新 fake を使うとカバー率が上がる
                    noise = torch.randn(batch_size, LATENT_DIM, device=device)
                    fake_specs = generator(noise, word_vecs)
                    outputs = discriminator(
                        add_instance_noise(fake_specs, 0.05), word_vecs
                    )
                    g_loss = criterion(outputs, real_labels)
                    g_loss.backward()
                    optimizer_g.step()

                # learn generator
                optimizer_g.zero_grad()

                outputs = discriminator(fake_specs, word_vecs)
                g_loss = criterion(outputs, real_labels)

                g_loss.backward()
                optimizer_g.step()

                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()

                dl += 1

                if (dl + 1) % 10 == 0:
                    print(
                        f"  Batch [{dl + 1}] D-Loss: {d_loss_sum / dl:.4f}, G-Loss: {g_loss_sum / dl:.4f}"
                    )

            print(
                f"Epoch [{epoch + 1}/{num_epochs}], D-Loss: {d_loss_sum / dl:.4f}, G-Loss: {g_loss_sum / dl:.4f}"
            )

            d_loss_list.append(d_loss_sum / dl)
            g_loss_list.append(g_loss_sum / dl)

            # --- 6. Save Checkpoint periodically ---
            if (epoch + 1) % 100 == 0:  # Save every 100 epochs
                print(f"Saving checkpoint at epoch {epoch + 1}...")
                checkpoint = {
                    "epoch": epoch,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "g_loss": g_loss,
                    "d_loss": d_loss,
                }
                save_checkpoint(epoch + start_epoch, checkpoint, CHECKPOINT_FOLDER)
                cleanup_checkpoints(CHECKPOINT_FOLDER, max_checkpoints=5)
        except KeyboardInterrupt:
            return generator, discriminator, d_loss_list, g_loss_list

    return generator, discriminator, d_loss_list, g_loss_list


def save_model(generator, discriminator):
    # Save models
    now = datetime.datetime.now()
    now = now.strftime("%Y:%m:%d-%H:%M:%S")
    torch.save(generator.state_dict(), "models/generator/latest.pth")
    torch.save(generator.state_dict(), f"models/generator/{now}.pth")
    torch.save(discriminator.state_dict(), "models/discriminator/latest.pth")
    torch.save(discriminator.state_dict(), f"models/discriminator/{now}.pth")
    cleanup_models("models/generator", max_models=10)
    cleanup_models("models/discriminator", max_models=10)
    print("Models saved.")


def save_plot(d_loss_list, g_loss_list):
    plt.plot(d_loss_list)
    plt.plot(g_loss_list)
    plt.savefig("loss.png")


if __name__ == "__main__":
    (
        device,
        dataloader,
        generator,
        discriminator,
        criterion,
        optimizer_g,
        optimizer_d,
    ) = setup()
    try:
        generator, discriminator, d_loss_list, g_loss_list = run(
            device,
            dataloader,
            generator,
            discriminator,
            optimizer_d,
            optimizer_g,
            criterion,
        )
    except KeyboardInterrupt:
        ans = input("Save model?(y/n)")
        if ans == "y":
            save_model(generator, discriminator)
            save_plot(d_loss_list, g_loss_list)
        else:
            pass

    save_model(generator, discriminator)
    save_plot(d_loss_list, g_loss_list)
