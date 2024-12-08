# Capsule Network App

The **Capsule Network App** is a user-friendly tool designed to help users understand, train, test, and perform inference with Capsule Networks, a cutting-edge deep learning architecture introduced by Geoffrey Hinton.

---

## 📚 Overview

The app offers the following features:
- Interactive web-based UI built with **Streamlit**.
- Options to **Train**, **Test**, and **Infer** Capsule Network models.
- Detailed explanations and visual aids for better understanding of Capsule Networks.
- Tools to upload images for inference and customize training parameters.

---

## 🚀 Features

- **Interactive UI**: User-friendly interface built with Streamlit.
- **Training Interface**: Customize and train your own Capsule Network models.
- **Testing Module**: Evaluate model performance with various metrics.
- **Inference Tool**: Perform predictions on uploaded images.
- **Visual Aids**: Illustrative images and diagrams to enhance understanding.

---

## 🖥️ Directory Structure

```plaintext
Capsule-Network-App/
├── app.py
├── CapsuleLayers.py
├── images/
│   ├── architecture_capsnet.png
│   ├── working_capsnet.png
│   └── header_image.png
├── requirements.txt
└── README.md
```

### File Details

- **`app.py`**: Main application script.
- **`images/`**: Contains visual aids (diagrams and header image).
- **`requirements.txt`**: List of Python dependencies for the app.
- **`README.md`**: Instructions and details about the project.

## ⚙️ Installation

### Prerequisites

- **Python 3.9 or higher**: [Download here](https://www.python.org/downloads/).
- **Git**: [Download here](https://git-scm.com/downloads).

### Steps to Install

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Capsule-Network-App.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd Capsule-Network-App
    ```

3. **(Optional) Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```

4. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application**:
    ```bash
    streamlit run app.py
    ```


## 🛠️ Training, Testing, and Inference

### 🏋️ Train
- Customize parameters like batch size, epochs, and learning rate.
- Start training by clicking the **Start Training** button.

### 🧪 Test
- Load a trained model and select a test dataset.
- Evaluate performance with metrics like accuracy, precision, and F1-score.

### 🔍 Infer
- Upload an image, and the app will preprocess it and display predictions with confidence scores.

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. **Fork the repository**.

2. **Create a new branch**:
    ```bash
    git checkout -b feature/YourFeatureName
    ```

3. **Commit your changes**:
    ```bash
    git commit -m "Add YourFeatureName"
    ```

4. **Push to your branch**:
    ```bash
    git push origin feature/YourFeatureName
    ```

5. **Open a pull request**.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙏 Acknowledgments

- **Geoffrey Hinton** for introducing Capsule Networks.
- **Streamlit** for the interactive web app framework.
- **Perplexity** for AI research.

---

© 2024 Capsule Network App. All rights reserved.
