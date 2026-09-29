import argparse
import os

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from model import SRGenerator


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        default="results"
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4
    )

    args = parser.parse_args()

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SRGenerator(
        in_channels=3,
        base_channels=32,
        num_res_blocks=6,
        scale=args.scale
    )

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    image = Image.open(
        args.input
    ).convert("RGB")

    tensor = TF.to_tensor(
        image
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        output = model(
            tensor
        )

    output = output.squeeze(0)
    output = output.clamp(0, 1)
    output = output.cpu()

    sr_image = TF.to_pil_image(
        output
    )

    output_path = os.path.join(
        args.out_dir,
        "enhanced_image.png"
    )

    sr_image.save(
        output_path
    )

    print()
    print(
        "Input:",
        image.size
    )

    print(
        "Output:",
        sr_image.size
    )

    print()
    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()