Read the **Wildfire Detection with NASA FIRMS Data** file for project details.

# NASA Wildfire Detection & Regression Project

This project contains **machine learning models** for predicting wildfires using satellite data:

1. **Classification Model** – Predicts the fire **type**.
2. **Regression Model** – Predicts **Fire Radiative Power (FRP)**.

Both models are built using **TensorFlow/Keras** and deployed on **Google Cloud Vertex AI** via **CI/CD pipelines**.

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Data](#data)  
- [Environment Setup](#environment-setup)  
- [Training](#training)  
- [Deployment](#deployment)  
- [CI/CD Pipeline](#cicd-pipeline)  
- [API Usage](#api-usage)  
- [Permissions](#permissions)  

---

## Project Structure

## Project Structure

Capstone/

├─ data/                     # Input datasets (e.g., NASA.csv)

├─ src-class/                # Classification model source code

│   ├─ train_class.py
│   └─ serve_class.py
├─ src-reg/                  # Regression model source code
│   ├─ train_reg.py
│   └─ serve_reg.py
├─ Dockerfile.train_class
├─ Dockerfile.serve_class
├─ Dockerfile.train_reg
├─ Dockerfile.serve_reg
├─ cloudbuild-classification.yaml
├─ cloudbuild-regression.yaml
├─ requirements.txt
└─ README.md


## Environment Setup

1. Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
2. Install Google Cloud SDK and authenticate:
```bash
gcloud auth login
gcloud config set project capstoneproject-462618
```


