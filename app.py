import os

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from model import SRGenerator


st.set_page_config(
    page_title="Mars Super Resolution",
    page_icon="🛰️",
    layout="wide"
)


st.title("🛰️ Mars Image Super Resolution")

st.write(
    "Upload a Mars or satellite image and enhance it using deep learning."
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


st.info(
    f"Device: {device}"
)


@st.cache_resource
def load_model():

    model = SRGenerator(
        in_channels=3,
        base_channels=32,
        num_res_blocks=6,
        scale=4
    )

    checkpoint = torch.load(
        "checkpoints/best.pth",
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    model.eval()

    return model


try:

    model = load_model()

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


uploaded_file = st.file_uploader(
    "Upload Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "tif",
        "tiff"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original"
        )

        st.image(
            image,
            use_container_width=True
        )

        st.write(
            f"Size: {image.width} × {image.height}"
        )


    if st.button(
        "🚀 Enhance Image",
        use_container_width=True
    ):

        with st.spinner(
            "AI is enhancing the image..."
        ):

            max_size = 256

            width, height = image.size

            if max(width, height) > max_size:

                ratio = (
                    max_size /
                    max(width, height)
                )

                new_width = int(
                    width * ratio
                )

                new_height = int(
                    height * ratio
                )

                small_image = image.resize(
                    (
                        new_width,
                        new_height
                    ),
                    Image.Resampling.LANCZOS
                )

            else:

                small_image = image


            tensor = ToTensor()(
                small_image
            )

            tensor = tensor.unsqueeze(
                0
            ).to(device)


            with torch.inference_mode():

                output = model(
                    tensor
                )


            output = output.squeeze(
                0
            )

            output = output.clamp(
                0,
                1
            )

            output = output.cpu()


            sr_image = ToPILImage()(
                output
            )


        with col2:

            st.subheader(
                "AI Enhanced"
            )

            st.image(
                sr_image,
                use_container_width=True
            )

            st.write(
                f"Size: {sr_image.width} × {sr_image.height}"
            )


        os.makedirs(
            "results",
            exist_ok=True
        )

        output_path = (
            "results/"
            "streamlit_output.png"
        )

        sr_image.save(
            output_path
        )


        st.success(
            "Enhancement completed!"
        )


        with open(
            output_path,
            "rb"
        ) as file:

            st.download_button(
                "📥 Download Enhanced Image",
                file,
                file_name="mars_enhanced.png",
                mime="image/png",
                use_container_width=True
            )