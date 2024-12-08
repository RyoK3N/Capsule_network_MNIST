# pages/3_Inference.py

import streamlit as st
import torch
from torch import nn
from torchvision import transforms, datasets
from CapsuleLayers import DenseCapsule, PrimaryCapsule
import os
import csv
from time import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ----------------------------
# Define Essential Functions
# ----------------------------

def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs

def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    L = y_true * torch.clamp(0.9 - y_pred, min=0.)**2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.)**2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon, x)
    return L_margin + lam_recon * L_recon

class CapsuleNet(nn.Module):
    def __init__(self, input_size, classes, routings, primary_num_maps=32, primary_num_dims=8, digit_num_dims=16):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)
        self.primarycaps = PrimaryCapsule(num_maps=primary_num_maps, num_dims=primary_num_dims)
        self.digitcaps = DenseCapsule(num_caps_in=primary_num_maps * 6 * 6, num_dims_in=primary_num_dims,
                                      num_caps_out=classes, num_dims_out=digit_num_dims, routings=routings)
        self.decoder = nn.Sequential(
            nn.Linear(digit_num_dims * classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        out_conv = self.relu(self.conv1(x))
        out_primary = self.primarycaps(out_conv)
        out_digit = self.digitcaps(out_primary)
        length = out_digit.norm(dim=-1)
        if y is None:
            index = length.max(dim=1)[1]
            y = torch.zeros(length.size(), device=x.device)
            y.scatter_(1, index.view(-1,1), 1.0)
        reconstruction = self.decoder((out_digit * y[:, :, None]).view(out_digit.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size), out_conv, out_primary, out_digit

@st.cache_data
def load_mnist(path='./data', download=True, batch_size=100, shift_pixels=2):
    kwargs = {'num_workers':4, 'pin_memory':True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=download,
                       transform=transforms.Compose([
                           transforms.RandomCrop(size=28, padding=shift_pixels),
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=download,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    return train_loader, test_loader

def inference(model, image, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])
    with torch.no_grad():
        img = transform(image).unsqueeze(0).to(device)
        y_pred, x_recon, _, _, _ = model(img)
        pred_class = y_pred.argmax(dim=1).item()
    return pred_class, x_recon.squeeze(0).cpu().numpy()

def main():
    st.title("Capsule Network Inference")

    # Define inference parameters
    st.sidebar.header("Inference Parameters")
    data_dir = st.sidebar.text_input("Data Directory", value="./data")
    save_dir = st.sidebar.text_input("Save Directory", value="./result")

    args = {
        "data_dir": data_dir,
        "save_dir": save_dir,
        "lam_recon": 0.392  # Must match training
    }

    # Helper function to list available models
    def get_available_models(save_dir):
        if not os.path.exists(save_dir):
            return []
        models = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
        return models

    available_models = get_available_models(args["save_dir"])
    if available_models:
        selected_model = st.selectbox("Select a model for inference:", available_models)
        uploaded_file = st.file_uploader("Upload an image for inference", type=["png", "jpg", "jpeg"])

        if st.button("Run Inference"):
            if uploaded_file is not None:
                try:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    st.write(f"Using device: {device}")

                    # Initialize model
                    model = CapsuleNet(
                        input_size=[1,28,28],
                        classes=10,
                        routings=3,  # Must match training
                        primary_num_maps=32,  # Must match training
                        primary_num_dims=8,  # Must match training
                        digit_num_dims=16  # Must match training
                    ).to(device)

                    # Load model weights
                    model_path = os.path.join(args["save_dir"], selected_model)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    st.success(f"Model {selected_model} loaded successfully.")

                    # Process uploaded image
                    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
                    st.image(image, caption='Uploaded Image', use_column_width=True)

                    # Run inference
                    with st.spinner('Performing inference...'):
                        pred_class, x_recon = inference(model, image, device)
                    st.success("Inference completed.")
                    st.write(f"**Predicted Class:** {pred_class}")

                    # Show reconstruction
                    reconstructed_image = x_recon.squeeze()
                    fig, axs = plt.subplots(1,2, figsize=(8,4))
                    axs[0].imshow(image.resize((28,28)), cmap='gray')
                    axs[0].axis('off')
                    axs[0].set_title("Original Image")

                    axs[1].imshow(reconstructed_image, cmap='gray')
                    axs[1].axis('off')
                    axs[1].set_title("Reconstructed Image")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"An error occurred during inference: {e}")
            else:
                st.warning("Please upload an image for inference.")
    else:
        st.info("No trained models found in the save directory. Please train a model first.")

if __name__ == "__main__":
    main()