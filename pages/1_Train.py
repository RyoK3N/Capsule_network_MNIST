# pages/1_Train.py

import streamlit as st
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torchvision import transforms, datasets
from CapsuleLayers import DenseCapsule, PrimaryCapsule, CapsuleNet
import os
import csv
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

def combine_images(generated_images):
    """
    Combine a batch of images into a grid for visualization.
    """
    num = generated_images.shape[0]
    width = int(np.sqrt(num))
    height = int(np.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((int(height*shape[0]), int(width*shape[1]), generated_images.shape[3]),
                     dtype=generated_images.dtype)
    for i, img in enumerate(generated_images):
        row = i // width
        col = i % width
        image[row*shape[0]: (row+1)*shape[0],
              col*shape[1]: (col+1)*shape[1], :] = img
    return image

def show_reconstruction(model, test_loader, n_images, save_dir, device):
    """
    Display and save a batch of original and reconstructed images.
    """
    model.eval()
    with torch.no_grad():
        for x, _ in test_loader:
            x = x[:min(n_images, x.size(0))].to(device)
            y_pred, x_recon, _, _, _ = model(x)
            data = torch.cat([x.cpu(), x_recon.cpu()], dim=0).numpy()
            img = combine_images(np.transpose(data, [0,2,3,1]))
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(save_dir, "real_and_recon.png"))
            st.write(f'Reconstructed images saved to {save_dir}/real_and_recon.png')
            st.image(img, use_container_width=True)
            break

def test_model(model, test_loader, args, device):
    """
    Evaluate the model on the entire test set.
    Returns:
        test_loss (float): Average loss over the test set.
        accuracy (float): Accuracy over the test set.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_onehot = torch.zeros(y.size(0), 10, device=device)
            y_onehot.scatter_(1, y.view(-1,1), 1.)
            y_pred, x_recon, _, _, _ = model(x, y_onehot)
            loss = caps_loss(y_onehot, y_pred, x, x_recon, args["lam_recon"])
            test_loss += loss.item() * x.size(0)
            pred = y_pred.data.max(1)[1]
            correct += pred.eq(y).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

def visualize_activations_batch(x, x_recon, out_conv, out_primary, out_digit,
                                images_placeholder, conv_placeholder, primary_digit_placeholder, primary_num_maps):
    """
    Visualize activations of different layers for a batch.
    Displays:
        - Original & Reconstructed Images
        - conv1 Feature Maps
        - PrimaryCaps Norm Heatmap and DigitCaps Norm Bar Chart side by side
    """
    x_np = x.detach().cpu().numpy()
    x_recon_np = x_recon.detach().cpu().numpy()
    out_conv_np = out_conv.detach().cpu().numpy()
    out_primary_np = out_primary.detach().cpu().numpy()
    out_digit_np = out_digit.detach().cpu().numpy()

    num_display = min(5, x.shape[0])

    # Original & Reconstructed Images
    fig1, axs1 = plt.subplots(2, num_display, figsize=(10,2))
    for i in range(num_display):
        axs1[0,i].imshow(x_np[i,0], cmap='gray')
        axs1[0,i].axis('off')
        axs1[0,i].set_title("Original", fontsize=6)

        axs1[1,i].imshow(x_recon_np[i,0], cmap='gray')
        axs1[1,i].axis('off')
        axs1[1,i].set_title("Recon", fontsize=6)
    plt.tight_layout()
    images_placeholder.pyplot(fig1)
    plt.close(fig1)

    # conv1 Feature Maps
    conv_maps = out_conv_np[0]
    num_conv_maps = min(8, conv_maps.shape[0])
    fig2, axs2 = plt.subplots(1, num_conv_maps, figsize=(1.5*num_conv_maps,1.5))
    for i in range(num_conv_maps):
        axs2[i].imshow(conv_maps[i], cmap='gray')
        axs2[i].axis('off')
    fig2.suptitle("conv1 Feature Maps", fontsize=8)
    plt.tight_layout()
    conv_placeholder.pyplot(fig2)
    plt.close(fig2)

    # PrimaryCaps Norm Heatmap and DigitCaps Norm Bar Chart side by side
    primary_caps_norm = np.linalg.norm(out_primary_np[0], axis=-1)
    primary_caps_norm = primary_caps_norm.reshape(6,6,primary_num_maps)
    primary_caps_map = np.mean(primary_caps_norm, axis=2)

    digit_caps_norm = np.linalg.norm(out_digit_np[0], axis=-1)

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(3,2))
    
    # PrimaryCaps Norm Heatmap
    cax = ax3a.imshow(primary_caps_map, cmap='jet')
    ax3a.set_title("PrimaryCaps Norm", fontsize=6)
    ax3a.axis('off')
    fig3.colorbar(cax, ax=ax3a, shrink=0.5, pad=0.02)

    # DigitCaps Norm Bar Chart
    ax3b.bar(range(10), digit_caps_norm, color='skyblue')
    ax3b.set_title("DigitCaps Norm", fontsize=6)
    ax3b.set_xlabel("Class", fontsize=5)
    ax3b.set_ylabel("Norm", fontsize=5)
    ax3b.tick_params(axis='both', which='major', labelsize=5)
    plt.tight_layout()
    primary_digit_placeholder.pyplot(fig3)
    plt.close(fig3)

def train_model(model, train_loader, test_loader, args, device):
    """
    Train the Capsule Network model.
    """
    st.write('Begin Training'+'-'*70)
    logfile_path = os.path.join(args["save_dir"], 'log.csv')
    with open(logfile_path, 'w', newline='') as logfile:
        logwriter = csv.DictWriter(logfile, fieldnames=['epoch','loss','val_loss','val_acc'])
        logwriter.writeheader()

        t0 = time()
        optimizer = Adam(model.parameters(), lr=args["lr"])
        lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args["lr_decay"])
        best_val_acc = 0.

        # Placeholders for progress and stats
        epoch_progress = st.progress(0)
        batch_progress = st.progress(0)
        epoch_info_placeholder = st.empty()
        time_info_placeholder = st.empty()

        st.subheader("Training Visualizations")
        images_placeholder = st.empty()
        conv_placeholder = st.empty()
        primary_digit_placeholder = st.empty()

        total_epochs = args["epochs"]
        for epoch in range(total_epochs):
            model.train()
            training_loss = 0.0
            num_batches = len(train_loader)
            batch_count = 0
            epoch_start_time = time()

            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                y_onehot = torch.zeros(y.size(0), 10, device=device)
                y_onehot.scatter_(1, y.view(-1,1), 1.)

                optimizer.zero_grad()
                y_pred, x_recon, out_conv, out_primary, out_digit = model(x, y_onehot)
                loss = caps_loss(y_onehot, y_pred, x, x_recon, args["lam_recon"])
                loss.backward()
                training_loss += loss.item() * x.size(0)
                optimizer.step()

                batch_count += 1
                batch_progress.progress(int((batch_count / num_batches) * 100))

                # Visualization after each batch
                visualize_activations_batch(
                    x, x_recon, out_conv, out_primary, out_digit,
                    images_placeholder, conv_placeholder, primary_digit_placeholder,
                    args["primary_num_maps"]
                )

            val_loss, val_acc = test_model(model, test_loader, args, device)
            lr_decay.step()

            # Compute epoch time and ETA
            epoch_time = time() - epoch_start_time
            avg_epoch_time = (time() - t0) / (epoch + 1)
            remaining_epochs = total_epochs - (epoch + 1)
            eta = remaining_epochs * avg_epoch_time

            logwriter.writerow({
                'epoch': epoch + 1,
                'loss': training_loss / len(train_loader.dataset),
                'val_loss': val_loss,
                'val_acc': val_acc
            })

            # Update placeholders with timing and epoch info
            epoch_info_placeholder.markdown(f"**Epoch:** {epoch+1}/{total_epochs} | **Val Acc:** {val_acc:.4f}")
            time_info_placeholder.markdown(
                f"**Epoch Time:** {epoch_time:.2f}s | **Estimated Remaining Time:** {eta:.2f}s"
            )

            st.write(f"==> Epoch {epoch+1}/{total_epochs}: loss={training_loss/len(train_loader.dataset):.5f}, "
                     f"val_loss={val_loss:.5f}, val_acc={val_acc:.4f}, epoch_time={epoch_time:.2f}s, ETA={eta:.2f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = os.path.join(args["save_dir"], f'epoch{epoch+1}.pkl')
                torch.save(model.state_dict(), best_model_path)
                st.write(f"**Best val_acc increased to {best_val_acc:.4f}**. Saved model to {best_model_path}")

            epoch_progress.progress(int(((epoch + 1) / total_epochs) * 100))

def main():
    st.title("Train Capsule Network Model")

    # Define training parameters
    st.sidebar.header("Training Parameters")
    primary_num_maps = st.sidebar.select_slider(
        "Number of PrimaryCapsule Maps",
        options=[16, 32, 48, 64],
        value=32
    )
    primary_num_dims = st.sidebar.select_slider(
        "Number of PrimaryCapsule Dimensions",
        options=[4, 8, 12, 16],
        value=8
    )
    digit_num_dims = st.sidebar.select_slider(
        "Number of DigitCapsule Dimensions",
        options=[8, 16, 24, 32],
        value=16
    )
    routings = st.sidebar.slider(
        "Number of Routing Iterations",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )
    epochs = st.sidebar.slider(
        "Number of Training Epochs",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=16,
        max_value=512,
        value=100,
        step=1,
        format="%d"
    )
    learning_rate = st.sidebar.number_input(
        "Learning Rate",
        min_value=1e-5,
        max_value=1e-2,
        value=0.001,
        step=1e-4,
        format="%.5f"
    )
    lr_decay = st.sidebar.slider(
        "Learning Rate Decay (gamma)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.1
    )
    lam_recon = st.sidebar.number_input(
        "Reconstruction Loss Lambda",
        min_value=1e-6,
        max_value=1.0,
        value=0.392,  # Example value, adjust as needed
        step=1e-4,
        format="%.6f"
    )

    # Update args dictionary with user inputs
    args = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(learning_rate),
        "lr_decay": float(lr_decay),
        "lam_recon": float(lam_recon),
        "routings": int(routings),
        "shift_pixels": 2,
        "data_dir": "./MNIST",  # Ensure this matches your dataset path
        "save_dir": "./result",
        "weights": None,
        "testing": False,
        "primary_num_maps": int(primary_num_maps),
        "primary_num_dims": int(primary_num_dims),
        "digit_num_dims": int(digit_num_dims)
    }

    # Create save directory if it doesn't exist
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    # Initialize session state variables
    if 'model' not in st.session_state:
        st.session_state['model'] = None
    if 'train_loader' not in st.session_state:
        st.session_state['train_loader'] = None
    if 'test_loader' not in st.session_state:
        st.session_state['test_loader'] = None
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False

    # Start training
    with st.expander("Training Options"):
        if not st.session_state['trained']:
            if st.button("Start Training"):
                try:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    st.write(f"Using device: {device}")

                    with st.spinner('Loading data...'):
                        train_loader, test_loader = load_mnist(args["data_dir"], download=True, batch_size=args["batch_size"])
                        st.session_state['train_loader'] = train_loader
                        st.session_state['test_loader'] = test_loader
                    st.success("Data loaded.")

                    with st.spinner('Initializing model...'):
                        model = CapsuleNet(
                            input_size=[1,28,28],
                            classes=10,
                            routings=args["routings"],
                            primary_num_maps=args["primary_num_maps"],
                            primary_num_dims=args["primary_num_dims"],
                            digit_num_dims=args["digit_num_dims"]
                        ).to(device)
                        st.session_state['model'] = model
                        st.write(model)
                    st.success("Model initialized.")

                    with st.spinner('Training in progress...'):
                        train_model(model, train_loader, test_loader, args, device)
                        st.session_state['trained'] = True
                    st.success("Training completed.")

                    with st.spinner('Showing reconstructions...'):
                        show_reconstruction(model, test_loader, 10, args["save_dir"], device)
                    st.success("Reconstruction displayed.")
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
        else:
            st.info("Model has been trained. You can proceed to testing or inference.")

    st.markdown("---")
    st.write("**Notes:**")
    st.write("""
    - **Training:** Click on "Start Training" to begin training the Capsule Network. Training progress, including loss and accuracy, will be displayed in real-time.
    - **Model Saving:** Trained models are saved in the `./result` directory with filenames like `epoch{epoch_number}.pkl`. The best performing model based on validation accuracy is also saved.
    - **Visualization:** The training visualizations include original vs. reconstructed images, convolutional feature maps, PrimaryCapsule norms, and DigitCapsule norms.
    - **Exception Handling:** The application handles common errors gracefully and provides informative messages to guide the user.
    - **Ablation Studies:** Adjust the number of PrimaryCapsule maps, dimensions, and routing iterations to perform ablation studies and observe their impact on model performance.
    """)

if __name__ == "__main__":
    main()