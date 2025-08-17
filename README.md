# Capstone Project: Wildfire Classification & Regression Models

## Project Overview
This project implements two machine learning pipelines for wildfire detection and intensity prediction using NASA satellite data:

1. **Classification Model** – Predicts wildfire occurrence.
2. **Regression Model** – Predicts Fire Radiative Power (FRP) for intensity estimation.

Both models are trained and deployed using **Google Cloud Platform (GCP)** with Dockerized pipelines and Vertex AI endpoints. CI/CD is automated with **Cloud Build**, so any push to GitHub triggers training and deployment automatically.

---

## Project Structure

```text
Capstone/
    data/                     # Input datasets (e.g., NASA.csv)
    src-class/                # Classification model source code
        train_class.py
        serve_class.py
    src-reg/                  # Regression model source code
        train_reg.py
        serve_reg.py
    Dockerfile.train_class
    Dockerfile.serve_class
    Dockerfile.train_reg
    Dockerfile.serve_reg
    cloudbuild-classification.yaml
    cloudbuild-regression.yaml
    requirements.txt
    README.md
```

---
## Pipeline Architecture

<img width="486" height="500" alt="image" src="https://github.com/user-attachments/assets/c2fb1f48-cff7-4e56-8a3b-584a7a853b3c" />

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/Capstone.git
cd Capstone
```

### 2. Configure GCP
- Set the project:
```bash
gcloud config set project <YOUR_PROJECT_ID>
```
- Enable required APIs:
```bash
gcloud services enable aiplatform.googleapis.com cloudbuild.googleapis.com storage.googleapis.com
```
- Create a GCS bucket for storing model artifacts:
```bash
gsutil mb -l us-central1 gs://<YOUR_BUCKET_NAME>
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Use

### Training & Deployment
Training and deployment are fully automated via Cloud Build pipelines. Simply **push any changes to GitHub**, and the following steps are triggered automatically:

1. Docker images for training are built and pushed.
2. Vertex AI custom training jobs are started.
3. Trained models are saved to GCS under the appropriate prefix (`ClassModel` or `RegModel`).
4. Docker images for serving are built and pushed.
5. Models are registered in Vertex AI Model Registry.
6. Models are deployed to Vertex AI endpoints:
   - Classification endpoint: `classification-prediction-endpoint`
   - Regression endpoint: `regression-prediction-endpoint`

### Testing the Endpoint
Send a POST request with JSON payload:

#### Classification Example:
```json
{
  "instances": [
    [68.41, 83.61, 329.46, 247.02, 3.61 ,0.58, 0.7, 1, 2]     
  ]
}
```

#### Regression Example:
```json
{
  "instances": [
    [34.5, -117.3, 310.7, 299.9, 1.2, 0.8, 0, 2]
  ]
}
```

**Note:** Use the appropriate endpoint for each model.

**OR**
## Use the UI
Classification Prediction Form
<img width="848" height="401" alt="image" src="https://github.com/user-attachments/assets/39099a07-aa8e-418e-ab0b-77e6cbf620ff" />

Regression Predection Form
<img width="975" height="461" alt="image" src="https://github.com/user-attachments/assets/e634deab-34dd-43f3-88fc-a6bdd5081f25" />

---

## Environment Variables for Serving
Set these in your serving container:

```bash
GCS_BUCKET=<YOUR_BUCKET_NAME>
GCS_PREFIX=ClassModel  # or RegModel
PORT=8080
```

---

## Permissions Needed on GCP
- `roles/storage.admin` – Read/write to GCS buckets
- `roles/aiplatform.admin` – Register and deploy models in Vertex AI
- `roles/cloudbuild.builds.editor` – Run Cloud Build pipelines
- `roles/iam.serviceAccountUser` – Execute jobs with service accounts

---

## Notes
- Any push to GitHub triggers the CI/CD pipeline automatically.
- Regression and classification models are deployed to **separate endpoints** to avoid overwriting artifacts.
- Ensure proper environment variables are set in serving containers to load models correctly.


