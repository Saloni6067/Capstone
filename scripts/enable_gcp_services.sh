#!/bin/bash

# Replace with your actual GCP project ID
PROJECT_ID="capstoneproject-462618"

# Set the GCP project
gcloud config set project "$PROJECT_ID"

# List of required GCP APIs to enable
APIS=(
  storage.googleapis.com
  cloudbuild.googleapis.com
  aiplatform.googleapis.com
)

# Enable each API one by one
for api in "${APIS[@]}"; do
  echo "Enabling $api..."
  gcloud services enable "$api"
done

echo "✅ All specified GCP services have been enabled."
