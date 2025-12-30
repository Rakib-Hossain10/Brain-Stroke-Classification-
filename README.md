# 🧠 Brain Stroke Classification System

A deep learning–based medical image classification system designed to identify **Bleeding**, **Ischemia**, and **Normal** brain conditions from medical images using a **ResNet50 convolutional neural network (CNN)**. The trained model is deployed as an interactive **Streamlit web application** for real-time inference and visualization.

---

##  Project Overview

Brain stroke is a critical medical condition where early detection is essential. This project explores the application of deep learning techniques to automatically classify brain scan images into three categories:
- **Bleeding**
- **Ischemia**
- **Normal**

The system follows an end-to-end machine learning workflow, including data preprocessing, CNN-based model training, performance evaluation, and cloud deployment.

---

##  Model Architecture

- Backbone: **ResNet50**
- Learning paradigm: **Transfer Learning**
- Output layer: **Softmax (3-class classification)**
- Loss function: **Categorical Cross-Entropy**
- Framework: **TensorFlow / Keras**

The model was fine-tuned on a labeled brain image dataset to learn discriminative features relevant to stroke classification.

---

## Key Features

- Multi-class classification (Bleeding / Ischemia / Normal)
- Real-time image upload and prediction
- Confidence score and probability distribution display
- Clean, professional UI built with Streamlit
- Optimized model loading using caching
- Cloud deployment via Streamlit Community Cloud

---

## Dataset & Preprocessing

- Input images resized to model-compatible dimensions
- RGB image normalization
- Central cropping and scaling
- Label mapping aligned with model output indices

*(Dataset details are omitted due to privacy and licensing constraints.)*

---

## 🚀 Deployment

The application is deployed as a web-based interface using **Streamlit**, allowing users to interact with the model directly through a browser.

### Live Application
 *  https://brain-stroke-classification-rakib-hossain.streamlit.app/

---


>  Large model files are managed using **Git LFS**.

---

##  Technologies Used

- **Python**
- **TensorFlow / Keras**
- **Convolutional Neural Networks (CNN)**
- **ResNet50**
- **Streamlit**
- **NumPy & Pandas**
- **Pillow (PIL)**
- **Git & GitHub**
- **Git LFS**
- **Streamlit Community Cloud**

---

## ▶️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Brain-Stroke-Classification.git
   cd Brain-Stroke-Classification/deployment


# Install dependencies:
-pip install -r requirements.txt

#Run the application:
-streamlit run app.py


**Disclaimer:** This project is developed for educational and research purposes only and must not be used for clinical diagnosis or medical decision-making.
