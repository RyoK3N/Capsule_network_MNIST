import streamlit as st
from PIL import Image
import os


base_img_path = '/Users/ryok3n/Desktop/Workspace/CapsNet/V3/images'
# Function to load images
def load_image(image_path):
    try:
        return Image.open(image_path)
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")
        return None

# Set up the page configuration
st.set_page_config(
    page_title="🎓 Capsule Network App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a catchy title with an image
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        "<h1 style='text-align: center; color: #FFA07A;'>🎓 Welcome to the Capsule Network App 🎓</h1>",
        unsafe_allow_html=True
    )
# with col2:
#     header_image_path = os.path.join("images", "drugs.gif")  # Replace with your header image filename
#     header_image = load_image(header_image_path)
#     if header_image:
#         st.image(
#             header_image,
#             use_container_width=True
#         )

# Introduction Section
st.header("🚀 Introduction")

st.markdown("""
Welcome to the **Capsule Network App**! This application empowers you to:

- **Train** a Capsule Network with customizable parameters.
- **Test** the performance of your trained models.
- **Perform Inference** to make predictions on new images.

Capsule Networks are a cutting-edge neural network architecture designed to better understand spatial hierarchies and relationships within data, overcoming some limitations of traditional Convolutional Neural Networks (CNNs). Let's embark on this learning journey together! 🧠✨
""")

# Divider
st.markdown("---")

# Theory Section with interactive dropdown
with st.expander("📚 Theory of Capsule Networks", expanded=False):
    st.markdown("""
    ### What are Capsule Networks?

    **Capsule Networks**, introduced by Geoffrey Hinton and his team, are designed to capture spatial hierarchies between different features in data. Unlike traditional CNNs that use scalar outputs, Capsule Networks utilize **vectors or tensors** to represent the properties of objects, enabling the network to recognize objects regardless of their orientation and position.
   """)
    #st.markdown("---")
    # Display Capsule Structure Image
    capsule_structure_path = os.path.join(base_img_path, "architecture_capsnet.png")
    capsule_structure_image = load_image(capsule_structure_path)
    if capsule_structure_image:
        st.image(
            capsule_structure_image,
            caption='Capsule Structure',
            use_container_width=True
        )
    st.markdown("""
    ### Key Components

    1. **Capsules**:
       - **Definition**: Groups of neurons whose activity vectors represent the instantiation parameters of specific entities (e.g., objects or object parts).
       - **Purpose**: Encode both the probability of an entity's existence and its various properties like pose, deformation, and velocity.

    2. **Dynamic Routing**:
       - **Definition**: A mechanism allowing lower-level capsules to send their outputs to appropriate higher-level capsules based on the agreement of their outputs.
       - **Purpose**: Ensures accurate mapping of lower-level features to higher-level entities.

    3. **Transformation Matrices**:
       - **Definition**: Learned matrices that transform the output of lower-level capsules to the viewpoint of higher-level capsules.
       - **Purpose**: Facilitate the prediction and routing process between capsules.

    ### Advantages Over Traditional CNNs

    - **Equivariance**: Maintains spatial relationships between features, allowing recognition of objects in various orientations and positions.
    - **Reduced Parameter Sharing**: Utilizes transformation matrices instead of extensive parameter sharing, minimizing redundancy.
    - **Robustness to Affine Transformations**: Better handles changes in viewpoint, scaling, and rotation.
    """)

# Working Principles Section with interactive dropdown
with st.expander("🔧 Working Principles of Capsule Networks", expanded=False):
    st.markdown("""
    ### 🏗️ Architecture Overview

    A typical Capsule Network architecture includes:
   """)
     # Display Capsule Structure Image
    capsule_structure_path = os.path.join(base_img_path, "working_capsnet.png")
    capsule_structure_image = load_image(capsule_structure_path)
    if capsule_structure_image:
        st.image(
            capsule_structure_image,
            caption='Capsule Structure',
            use_container_width=True
        )

    st.markdown("""   
                 
    1. **Convolutional Layer**:
       - Extracts low-level features from the input data.
       - Similar to traditional CNNs.

    2. **Primary Capsules Layer**:
       - Converts convolutional feature maps into capsules.
       - Each capsule outputs a vector representing various properties of detected features.

    3. **Digit Capsules Layer**:
       - Receives input from the Primary Capsules.
       - Utilizes dynamic routing to assign lower-level capsules to higher-level digit capsules.
       - Each Digit Capsule corresponds to a specific class (e.g., digits 0-9 in MNIST).

    4. **Reconstruction Subnetwork**:
       - Reconstructs the input image from the output of the Digit Capsules.
       - Acts as a regularizer to ensure that the capsules capture all necessary properties of the input.

    ### 🔄 Dynamic Routing Algorithm

    Dynamic routing determines how lower-level capsules send their outputs to higher-level capsules based on the agreement between their predictions.

    **Steps Involved:**
    """)
     # Display Dynamic Routing Image
    dynamic_routing_path = os.path.join(base_img_path, "dynamic_routing.png")  # Replace with the correct image if different
    dynamic_routing_image = load_image(dynamic_routing_path)
    if dynamic_routing_image:
        st.image(
            dynamic_routing_image,
            caption='Dynamic Routing',
            use_container_width=True
        )
    st.markdown("""
    1. **Prediction Vectors**:
       - Each lower-level capsule predicts the output of higher-level capsules using transformation matrices.
       $$
       \hat{\mathbf{u}}_{j|i} = \mathbf{W}_{ij} \mathbf{u}_i
       $$
       - Where:
         - $\mathbf{u}_i$ is the output vector of the lower-level capsule $i$.
         - $\mathbf{W}_{ij}$ is the transformation matrix from capsule $i$ to capsule $j$.
         - $\hat{\mathbf{u}}_{j|i}$ is the prediction vector for capsule $j$.

    2. **Routing by Agreement**:
       - Initialize the routing logits $b_{ij} = 0$.
       - Compute coupling coefficients using the softmax function:
         $$
         c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}
         $$
       - Update the output vectors of higher-level capsules:
         $$
         \mathbf{s}_j = \sum_i c_{ij} \hat{\mathbf{u}}_{j|i}
         $$
         $$
         \mathbf{v}_j = \text{Squash}(\mathbf{s}_j)
         $$
       - Update the routing logits based on the agreement:
         $$
         b_{ij} = b_{ij} + \hat{\mathbf{u}}_{j|i} \cdot \mathbf{v}_j
         $$
       - Repeat for a predefined number of iterations.

    3. **Squash Function**:
       - Ensures that the length of the output vector of a capsule is between 0 and 1, representing the probability of the entity's existence.
         $$
         \text{Squash}(\mathbf{s}) = \frac{||\mathbf{s}||^2}{1 + ||\mathbf{s}||^2} \frac{\mathbf{s}}{||\mathbf{s}||}
         $$
    """)


# Key Formulas Section with interactive dropdown
with st.expander("🧮 Key Formulas in Capsule Networks", expanded=False):
    st.markdown("""
    ### 1. **Squash Function**
    $$
    \text{Squash}(\mathbf{s}) = \frac{||\mathbf{s}||^2}{1 + ||\mathbf{s}||^2} \frac{\mathbf{s}}{||\mathbf{s}||}
    $$

    ### 2. **Margin Loss**
    $$
    L_k = T_k \max(0, m^+ - ||\mathbf{v}_k||)^2 + \lambda (1 - T_k) \max(0, ||\mathbf{v}_k|| - m^-)^2
    $$
    - Where:
      - $T_k$ is 1 if the class $k$ is present, else 0.
      - $m^+ = 0.9$ and $m^- = 0.1$ are the margin parameters.
      - $\lambda$ is a down-weighting factor for absent classes (commonly set to 0.5).

    ### 3. **Reconstruction Loss**
    $$
    L_{recon} = ||\mathbf{\hat{x}} - \mathbf{x}||^2
    $$
    - Where:
      - $\mathbf{\hat{x}}$ is the reconstructed image.
      - $\mathbf{x}$ is the original input image.

    ### 4. **Total Loss**
    $$
    L = \sum_k L_k + \beta L_{recon}
    $$
    - Where:
      - $\beta$ is a weighting factor (commonly set to 0.0005) to balance the reconstruction loss with the margin loss.

    ### 5. **Prediction Vector Calculation**
    $$
    \hat{\mathbf{u}}_{j|i} = \mathbf{W}_{ij} \mathbf{u}_i
    $$
    - As defined in the dynamic routing algorithm, where $\mathbf{W}_{ij}$ is the transformation matrix.

    ### 6. **Coupling Coefficients**
    $$
    c_{ij} = \frac{\exp(b_{ij})}{\sum_k \exp(b_{ik})}
    $$
    - Softmax function applied to the routing logits to obtain coupling coefficients.
    """)

# Divider
st.markdown("---")

# Training Section
with st.expander("🏋️‍♂️ Train the Capsule Network", expanded=False):
    st.markdown("""
    ### Training Parameters

    - **Epochs**: Number of times the entire dataset is passed through the network.
    - **Batch Size**: Number of samples processed before the model is updated.
    - **Learning Rate**: Step size at each iteration while moving toward a minimum of a loss function.
    - **Routing Iterations**: Number of times the dynamic routing algorithm runs.

    ### Training Process

    1. **Data Preparation**:
       - Load and preprocess the dataset.
       - Split into training and validation sets.

    2. **Model Initialization**:
       - Define the Capsule Network architecture.
       - Initialize weights and transformation matrices.

    3. **Training Loop**:
       - Forward pass: Compute predictions.
       - Compute loss using margin loss and reconstruction loss.
       - Backward pass: Update weights using an optimizer (e.g., Adam).
       - Monitor training and validation metrics.
    """)

# Testing Section
with st.expander("🧪 Test the Capsule Network", expanded=False):
    st.markdown("""
    ### Testing Parameters

    - **Test Dataset**: Dataset to evaluate the model's performance.
    - **Metrics**: Accuracy, Precision, Recall, F1-Score.

    ### Testing Process

    1. **Load Trained Model**:
       - Load the model weights saved during training.

    2. **Evaluate Performance**:
       - Run the model on the test dataset.
       - Compute evaluation metrics.
    """)

# Inference Section
with st.expander("🔍 Perform Inference", expanded=False):
    st.markdown("""
    ### Upload an Image for Prediction

    - **Supported Formats**: JPEG, PNG.
    - **Preprocessing**: Resize and normalize the image as per model requirements.

    ### Inference Process

    1. **Image Upload**:
       - Users can upload their own images for prediction.

    2. **Preprocessing**:
       - The app preprocesses the image to match the input shape expected by the model.

    3. **Prediction**:
       - The model outputs the predicted class and confidence score.

    """)
# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: gray;'>
    © 2024 Capsule Network App. All rights reserved.
    </p>
    """, unsafe_allow_html=True)