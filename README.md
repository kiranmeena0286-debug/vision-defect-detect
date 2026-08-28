# Industrial Vision Defect Detection 👁️🏭
<div align="center">nn[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=githubactions)](#)n[![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen?style=for-the-badge)](#)n[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue?style=for-the-badge&logo=python)](#)n[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#)n[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](#)nn</div>nn
A deep learning computer vision pipeline designed for real-time quality control in manufacturing. Uses a fine-tuned ResNet50 architecture to detect microscopic surface defects on metal casts (tying into metallurgical engineering contexts).

## Tech Stack
- **Framework:** PyTorch & TorchVision
- **Transforms:** OpenCV, Albumentations
- **Tracking:** Weights & Biases (W&B)
- **Deployment:** TorchServe & Docker

## Pipeline Features
- **Data Augmentation:** Heavy synthetic defect generation using geometric transformations and lighting shifts.
- **Transfer Learning:** Fine-tuned ResNet50 baseline with custom classification heads.
- **Evaluation:** Precision/Recall curves and Grad-CAM visualizations to explain model focus.
