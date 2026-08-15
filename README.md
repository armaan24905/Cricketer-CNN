# 🏏 Cricketer CNN — Image Classification System

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.21.0-orange?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Computer%20Vision-CNN-purple?style=for-the-badge" />

</p>

<p align="center">
  <b>🏏 Deep Learning • 💻 Computer Vision • 🖼️ Image Classification •🌍 CNN</b>
</p>

<p align="center"
An end-to-end Deep Learning system for classifying cricketer players from images using a Convolutional Neural Network.>

---

## 📌 Overview

**Cricketer CNN** is a end-to-end **Computer Vision** Deep Learning-based image classification system designed to identify cricket players from facial or portrait images.

The project uses a **Convolutional Neural Network (CNN)** implemented with **TensorFlow/Keras** to learn visual patterns from labeled cricket-player images and classify an input image into one of the supported player categories.

The project demonstrates a complete machine learning workflow:

> 📂 Dataset → 🧹 Dataset Validation → 🔄 Preprocessing → 🧠 CNN Architecture → 🚀 Training → 📊 Evaluation → 🔮 Prediction → 💾 Model Export

This project was developed as a practical **B.Tech Artificial Intelligence & Data Science / AI-ML engineering project** with a focus on understanding the complete computer vision pipeline rather than relying only on pre-trained models.

---
## 🏏 Currently Recognizes

The model is trained to classify images into 8 classes:

* AB de Villiers
* Brian Lara
* Other Cricketer
* Rahul Dravid
* Rohit Sharma
* Sachin Tendulkar
* Shane Warne
* Virat Kohli

---
## 📊 What the Project Provides

🖼️ Image-based cricketer classification
🧠 Custom CNN architecture
🔄 Image preprocessing and normalization
📈 Training and validation accuracy analysis
📉 Training and validation loss analysis
🔥 Confusion matrix
📋 Precision, Recall and F1-score
🎯 Prediction with confidence score
💾 Trained model export
🧹 Dataset validation for corrupted/invalid images

---

## 🎯 Main Objective

The main objective of this project is to demonstrate an end-to-end AI/ML Computer Vision workflow, from dataset preparation and model training to evaluation and real-world image prediction.

* Build a complete CNN-based image classification pipeline.
* Validate and prepare the image dataset.
* perform image resizing and normalization.
* Train a custom CNN using TensorFlow/Keras.
* Evaluate model performance using multiple metrics.
* Generate training and validation accuracy graphs.
* Analyze predictions using a confusion matrix.
* Calculate Precision, Recall and F1-Score.
* Export the trained model.
* perform predictions on unseen images.
* Provide confidence scores for prediction.
* Makes the project reproducible for other users.


---
## 🎯 Problem Statement

Identifying individuals from images is a common **Computer Vision Classification Problem.**

The objective of this project is to develop a CNN-based classification model capable of recognizing cricket players from their portrait or facial images.

Given an unseen image:

```text
Input Image
     ↓
Image Preprocessing
     ↓
CNN Feature Extraction
     ↓
Classification Layer
     ↓
Predicted Cricketer
     ↓
Confidence Score
