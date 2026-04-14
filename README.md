# 🧠 AI-Powered Medical Image Analysis System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/AI-Medical-Image-Analysis)](https://github.com/YOUR_USERNAME/AI-Medical-Image-Analysis/stargazers)

> **An AI-powered system that classifies Brain MRI images to detect tumors with 94% accuracy**

## 📋 Overview

This project implements a **Deep Learning system** for automated brain tumor detection from MRI scans. Using **Convolutional Neural Networks (CNN)** with **Transfer Learning (VGG16)**, the system classifies images into four categories:

| Class | Description | Color Code |
|-------|-------------|------------|
| 🧬 **Glioma Tumor** | A type of brain tumor that originates in glial cells | 🔴 Red |
| 🧫 **Meningioma Tumor** | Tumor that arises from the meninges (brain coverings) | 🟠 Orange |
| ✅ **No Tumor** | Healthy brain scan | 🟢 Green |
| 🧠 **Pituitary Tumor** | Tumor of the pituitary gland | 🟣 Purple |

## 🎯 Problem Statement

**The Challenge:** Manual analysis of brain MRI scans is:
- ⏰ **Time-consuming** (15-30 minutes per scan)
- 👨‍⚕️ **Expert-dependent** (requires specialized radiologists)
- 😓 **Prone to human error** (fatigue, oversight, inter-observer variability)

**Our Solution:** An AI system that provides:
- ⚡ **Instant analysis** (under 10 seconds)
- 🎯 **High accuracy** (94%+)
- 🔄 **Consistent results** (no fatigue)
- 🌐 **Accessible via web** (anywhere, anytime)

## 🏥 Industry Relevance

This technology is already being used by leading healthcare companies:

| Company | Application |
|---------|-------------|
| **Google DeepMind** | Medical imaging AI for disease detection |
| **IBM Watson Health** | Diagnostic assistance |
| **Siemens Healthineers** | AI-powered radiology |
| **Qure.ai** | Automated X-ray and CT analysis |
| **Butterfly Network** | Portable AI ultrasound |
| **Arterys** | Cloud-based medical imaging |

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Deep Learning** | TensorFlow 2.13, Keras |
| **Computer Vision** | OpenCV, PIL |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Model Architecture** | VGG16 (Transfer Learning) |

## 📊 Dataset

**Brain Tumor Classification (MRI)** from Kaggle

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Sartaj Bhuvaji](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) |
| **Total Images** | ~7,000 |
| **Classes** | 4 (Glioma, Meningioma, No Tumor, Pituitary) |
| **Format** | JPEG/PNG (pre-processed) |
| **Train/Test Split** | 80% / 20% |

### Class Distribution
![Class Distribution](images/class_distribution.png)

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Brain MRI Image │
│ (224×224×3) │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ PREPROCESSING │
│ • Resize to 224×224 │
│ • Normalize pixel values (0-1) │
│ • Data Augmentation (rotation, zoom, flip) │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION (VGG16 Base) │
│ • 13 Convolutional Layers │
│ • 5 Max Pooling Layers │
│ • Pre-trained on ImageNet │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ CLASSIFICATION HEAD │
│ • Global Average Pooling │
│ • Dense Layer (512 units + ReLU) + Dropout(0.5) │
│ • Dense Layer (256 units + ReLU) + Dropout(0.3) │
│ • Output Layer (4 units + Softmax) │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT │
│ Glioma Tumor | Meningioma Tumor | No Tumor | Pituitary Tumor │
│ + Confidence Score │
└─────────────────────────────────────────────────────────────────┘