# Industrial Vision Defect Detection 👁️🏭

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
