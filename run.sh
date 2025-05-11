#!/bin/bash
# run.sh
# This script executes the machine learning pipeline.

# Navigate to the 'src' directory (assuming run.sh is in the project root)
cd src

# Check if the data ingestion script exists
if [ ! -f "data_ingestion.py" ]; then
  echo "Error: data_ingestion.py not found in src directory."
  exit 1
fi

# Run data ingestion (replace with your database path and query)
echo "Running data_ingestion.py..."
python data_ingestion.py
echo "Data ingestion complete."

# Check if the data processing script exists
if [ ! -f "data_processing.py" ]; then
  echo "Error: data_processing.py not found in src directory."
  exit 1
fi

# Run data processing
echo "Running data_processing.py..."
python data_processing.py
echo "Data processing complete."

# Check if the feature engineering script exists
if [ ! -f "feature_engineering.py" ]; then
  echo "Error: feature_engineering.py not found in src directory."
  exit 1
fi
# Run feature engineering
echo "Running feature_engineering.py..."
python feature_engineering.py
echo "Feature engineering complete."

# Check if the model training script exists
if [ ! -f "model_training.py" ]; then
  echo "Error: model_training.py not found in src directory."
  exit 1
fi

# Run model training
echo "Running model_training.py..."
python model_training.py
echo "Model training complete."

echo "Pipeline execution finished."
