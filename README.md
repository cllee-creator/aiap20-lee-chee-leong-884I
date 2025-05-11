# aiap20-lee-chee-leong-884I
AIAP20 Technical Assessment Submission 

* **Full Name:** Lee Chee Leong
* **Email Address:** 77cllee@gmail.com

## Folder and File Structure

The project has the following folder structure:
aiap20-lee-chee-leong-884I/
├── .github/             # Contains GitHub Actions scripts (from the template)
├── src/                 # Contains the Python modules for the machine learning pipeline
│   ├── data_ingestion.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── eda.ipynb            # Jupyter Notebook containing the Exploratory Data Analysis
├── run.sh               # Bash script to execute the pipeline
├── requirements.txt     # List of Python packages required
└── README.md            # This file, explaining the pipeline and EDA summary

* `src/`: This directory contains the Python modules for the machine learning pipeline.
    * `data_ingestion.py`: Fetches/imports data from a source (e.g., SQLite database, CSV file).
    * `data_processing.py`: Performs basic data cleaning and preprocessing.
    * `feature_engineering.py`: Creates new features from existing data.
    * `model_training.py`: Trains the machine learning model(s).
    * `model_evaluation.py`: Evaluates the trained model(s).
* `eda.ipynb`: A Jupyter Notebook containing the Exploratory Data Analysis conducted in Task 1. This notebook includes visualizations and explanations of the data's characteristics and insights.
* `run.sh`: A bash script to execute the pipeline.
* `requirements.txt`: A list of Python packages required to run the pipeline.
* `README.md`: This file, providing an explanation of the pipeline design and its usage, and summarizing key findings from the EDA.

## Instructions for Executing the Pipeline

To run the pipeline:

1.  Ensure you have all the necessary dependencies installed. Navigate to the base directory of the project in your terminal and run:

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: This step might be handled automatically by GitHub Actions.)*

2.  Navigate to the base directory of the project in your terminal.

3.  Execute the `run.sh` script:

    ```bash
    bash run.sh
    ```

## Description of Logical Steps/Flow of the Pipeline

The pipeline consists of the following steps:

1.  **Data Ingestion:** The `data_ingestion.py` script loads data from a specified source (e.g., SQLite database, CSV file) into a pandas DataFrame.
2.  **Data Processing:** The `data_processing.py` script performs data cleaning and preprocessing, such as handling missing values, converting data types, and removing irrelevant columns.
3.  **Feature Engineering:** The `feature_engineering.py` script creates new features from the existing data, based on insights from the Exploratory Data Analysis (EDA) documented in `eda.ipynb`.
4.  **Model Training:** The `model_training.py` script trains one or more machine learning models on the processed data. The data is split into training and testing sets, and the model is trained on the training set.
5.  **Model Evaluation:** The `model_evaluation.py` script evaluates the performance of the trained model(s) on the testing set. It calculates and reports relevant evaluation metrics.

Here's a diagram illustrating the pipeline's flow:
Data Source --> Data Ingestion --> Data Processing --> Feature Engineering --> Model Training --> Model Evaluation --> Output (Metrics, Predictions)

## Overview of EDA Findings and Choices Made

The Exploratory Data Analysis (EDA) was conducted in the `eda.ipynb` notebook. Key findings from this analysis informed the design and implementation of the machine learning pipeline.

For example:

Key EDA Findings:

* The `column_A` has many missing values, which will be imputed with the median.
* The `column_B` is highly skewed, and a log transformation will be applied.
* `Column_C` and `Column_D` are strongly correlated, so `Column_D` will be dropped.
* The dataset has a class imbalance, which will be addressed during model training.
