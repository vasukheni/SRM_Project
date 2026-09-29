import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SRDataset, make_train_val_split
from model import SRGenerator


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints"
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=48
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=32
    )

    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=6
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002
    )

    parser.add_argument(
        "--samples_per_image",
        type=int,
        default=20
    )

    return parser.parse_args()


def calculate_psnr(sr, hr):

    mse = torch.mean(
        (sr - hr) ** 2
    )

    if mse.item() == 0:
        return 100.0

    return (
        10 *
        torch.log10(
            1.0 / mse
        )
    ).item()


def validate(model, loader, device):

    model.eval()

    total_loss = 0
    total_psnr = 0
    count = 0

    loss_function = nn.L1Loss()

    with torch.no_grad():

        for lr, hr in loader:

            lr = lr.to(device)
            hr = hr.to(device)

            sr = model(lr)

            loss = loss_function(
                sr,
                hr
            )

            psnr = calculate_psnr(
                sr,
                hr
            )

            total_loss += loss.item()
            total_psnr += psnr

            count += 1

    model.train()

    if count == 0:
        return 0, 0

    return (
        total_loss / count,
        total_psnr / count
    )


def main():

    args = get_args()

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)
    print()

    train_paths, val_paths = make_train_val_split(
        args.data_dir,
        0.1
    )

    train_dir = args.data_dir

    train_dataset = SRDataset(
        train_dir,
        patch_size=args.patch_size,
        scale=args.scale,
        augment=True,
        samples_per_image=args.samples_per_image
    )

    val_dataset = SRDataset(
        train_dir,
        patch_size=args.patch_size,
        scale=args.scale,
        augment=False,
        samples_per_image=5
    )

    train_dataset.paths = train_paths
    val_dataset.paths = val_paths

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        "Training images:",
        len(train_paths)
    )

    print(
        "Validation images:",
        len(val_paths)
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    model = SRGenerator(
        in_channels=3,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        scale=args.scale
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    loss_function = nn.L1Loss()

    best_psnr = 0

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        running_loss = 0

        for lr, hr in train_loader:

            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()

            sr = model(lr)

            loss = loss_function(
                sr,
                hr
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss /
            max(1, len(train_loader))
        )

        val_loss, val_psnr = validate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"| Loss: {train_loss:.5f} "
            f"| Val Loss: {val_loss:.5f} "
            f"| PSNR: {val_psnr:.2f} dB"
        )

        torch.save(
            model.state_dict(),
            os.path.join(
                args.out_dir,
                "last.pth"
            )
        )

        if val_psnr > best_psnr:

            best_psnr = val_psnr

            torch.save(
                model.state_dict(),
                os.path.join(
                    args.out_dir,
                    "best.pth"
                )
            )

            print(
                "Best model saved."
            )

    print()
    print(
        "Training completed."
    )

    print(
        "Best PSNR:",
        round(best_psnr, 2),
        "dB"
    )

    print(
        "Checkpoint:",
        args.out_dir
    )


if __name__ == "__main__":
    main()