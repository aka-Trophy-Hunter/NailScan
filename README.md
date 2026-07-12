# NailScan

An AI-powered nail health screening tool. Upload a photo of a nail and the app:

1. **Detects the nail region** using a YOLOv5 segmentation model
2. **Classifies the nail condition** using a ResNet-152 classifier

Live app: https://nailscan.streamlit.app/

## How it works

**Stage 1 — Nail detection/segmentation:** A YOLOv5 segmentation model locates the nail in the uploaded photo and draws a bounding box around it, confirming a nail is present before running classification.

**Stage 2 — Condition classification:** A ResNet-152 model (fine-tuned via transfer learning) classifies the nail into one of six categories:
- Healthy Nail
- Acral Lentiginous Melanoma
- Onychogryphosis
- Blue Finger
- Clubbing
- Pitting

## Model details

- **Classifier:** ResNet-152 (ImageNet-pretrained backbone, fine-tuned final layer), trained with standard augmentation (horizontal/vertical flip, rotation, color jitter) and class-weighted loss to handle class imbalance
- **Training data:** Public Kaggle nail disease dataset (~3,700 training images, ~90 validation images across the 6 classes above)
- **Segmentation:** YOLOv5 instance segmentation model for nail localization

## Tech stack

Python, PyTorch, torchvision, YOLOv5 (Ultralytics), Streamlit

## Disclaimer

This is an automated screening tool trained on a public dataset for demonstration purposes. It is **not a medical diagnostic tool**. Please consult a dermatologist or medical professional for any actual nail health concerns.
