# pages/2_Test.py

import streamlit as st
import torch
from torch import nn
from torchvision import transforms, datasets
from CapsuleLayers import DenseCapsule, PrimaryCapsule, CapsuleNet
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ----------------------------
# Define Essential Functions
# ----------------------------

def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule Network loss combining margin loss and reconstruction loss.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.)**2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.)**2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon, x)
    return L_margin + lam_recon * L_recon

@st.cache_data
def load_mnist(path='./data', download=True, batch_size=100, shift_pixels=2):
    """
    Load MNIST dataset with optional random cropping.
    """
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

def test_model_on_random_batch(model, test_loader, args, device):
    """
    Evaluate the model on a single random batch from the test_loader.
    Returns:
        test_loss (float): Loss on the batch.
        accuracy (float): Accuracy on the batch.
        images (torch.Tensor): Original images.
        x_recon (torch.Tensor): Reconstructed images.
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
    """
    model.eval()
    test_loss = 0
    correct = 0
    try:
        # Get a single batch
        x, y = next(iter(test_loader))
    except StopIteration:
        st.error("Test loader is empty.")
        return None, None, None, None, None, None

    x, y = x.to(device), y.to(device)
    y_onehot = torch.zeros(y.size(0), 10, device=device)
    y_onehot.scatter_(1, y.view(-1,1), 1.)

    with torch.no_grad():
        y_pred, x_recon, _, _, _ = model(x, y_onehot)
        loss = caps_loss(y_onehot, y_pred, x, x_recon, args["lam_recon"])
        test_loss += loss.item() * x.size(0)
        pred = y_pred.data.max(1)[1]
        correct += pred.eq(y).sum().item()

    test_loss /= len(x)
    accuracy = correct / len(x)

    return test_loss, accuracy, x.cpu(), x_recon.cpu(), y.cpu(), pred.cpu()

def display_batch(images, reconstructions, y_true, y_pred, num_images=10):
    """
    Display original and reconstructed images along with true and predicted labels.
    """
    num_images = min(num_images, images.shape[0])
    fig, axs = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i in range(num_images):
        # Original Images
        axs[0, i].imshow(images[i].squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title(f"True: {y_true[i]}")
        
        # Reconstructed Images
        axs[1, i].imshow(reconstructions[i].squeeze(), cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Pred: {y_pred[i]}")
    
    axs[0, 0].set_ylabel("Original", fontsize=12)
    axs[1, 0].set_ylabel("Reconstructed", fontsize=12)
    st.pyplot(fig)
    plt.close(fig)

def main():
    st.title("Test Capsule Network Model")

    # Define testing parameters
    st.sidebar.header("Testing Parameters")
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=16,
        max_value=512,
        value=100,
        step=1,
        format="%d"
    )
    data_dir = st.sidebar.text_input("Data Directory", value="./data")
    save_dir = st.sidebar.text_input("Save Directory", value="./result")

    args = {
        "batch_size": int(batch_size),
        "data_dir": data_dir,
        "save_dir": save_dir,
        "lam_recon": 0.392  # Ensure this matches training
    }

    # Helper function to list available models
    def get_available_models(save_dir):
        if not os.path.exists(save_dir):
            return []
        models = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
        return models

    available_models = get_available_models(args["save_dir"])
    if available_models:
        selected_model = st.selectbox("Select a model to test:", available_models)
        if st.button("Run Test"):
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
                if not os.path.exists(model_path):
                    st.error(f"Model file '{selected_model}' does not exist in '{args['save_dir']}'.")
                    return

                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                st.success(f"Model '{selected_model}' loaded successfully.")

                # Load data
                with st.spinner('Loading test data...'):
                    _, test_loader = load_mnist(args["data_dir"], download=True, batch_size=args["batch_size"])
                st.success("Test data loaded.")

                # Run test on a random batch
                with st.spinner('Testing model on a random batch...'):
                    test_loss, test_acc, images, reconstructions, y_true, y_pred = test_model_on_random_batch(model, test_loader, args, device)
                if test_loss is not None and test_acc is not None:
                    st.success("Testing completed.")
                    st.write(f"**Test Loss on Random Batch:** {test_loss:.5f}")
                    st.write(f"**Test Accuracy on Random Batch:** {test_acc:.4f}")

                    # Display original and reconstructed images with labels
                    st.subheader("Visualization of Test Batch")
                    display_batch(images, reconstructions, y_true, y_pred, num_images=10)
                else:
                    st.warning("Unable to compute test metrics.")
            except Exception as e:
                st.error(f"An error occurred during testing: {e}")
    else:
        st.info("No trained models found in the save directory. Please train a model first.")

    st.markdown("---")
    st.write("**Notes:**")
    st.write("""
    - **Testing:** Select a trained model and click "Run Test" to evaluate its performance on a random batch from the test dataset.
    - **Model Loading:** Ensure that the model architecture matches the saved model weights.
    - **Visualization:** After testing, a sample of original and reconstructed images will be displayed along with their true and predicted labels.
    - **Exception Handling:** The application handles common errors gracefully and provides informative messages to guide the user.
    """)

if __name__ == "__main__":
    main()