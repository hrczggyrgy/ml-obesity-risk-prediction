# Obesity Level Prediction

## Project Overview

This project aims to predict obesity levels using individual-specific data, such as body measurements, dietary habits, and family health history. We're exploring how these aspects contribute to obesity, with the goal of helping individuals and healthcare professionals better understand and manage this condition.

**Data Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster)

## Repository Structure

- **Datasets/**: Hosts the data files used in the project, including raw and processed datasets.
- **Notebooks/**: Features Jupyter notebooks for exploratory data analysis (EDA), model training, and evaluation. These notebooks detail our analysis methods and findings.
- **scripts/**: Contains Python scripts for data processing, model training, and other utilities.
- **Streamlit/**: Includes the setup for a Flask-based web app that offers real-time predictions based on the trained model.
- **pkls/**: Stores the serialized models and any data processing pipelines as pickle files.
- **.gitignore**: Specifies which files and directories Git should ignore.
- **LICENSE**: Describes the project's license and usage terms.
- **README.md**: Provides an overview of the project, setup instructions, and other essential information.
- **requirements.txt**: Lists the necessary Python packages to run the project.

## Data Details

The project's datasets are located in the `Datasets/` directory. The main dataset, `ObesityDataSet.csv`, includes a wide range of features from physical measurements to dietary habits and family disease histories. Additional derived datasets for training and testing purposes are also provided.

## Exploratory Data Analysis and Feature Engineering

The `EDA_for_obesity_risk_notebook.ipynb` notebook, found in the `Notebooks/` directory, dives deep into the dataset. It outlines the statistical tests used, methods for anomaly detection, and data visualizations that reveal insights about the data. Feature engineering efforts to enhance model performance are also documented here.

**Notebook Link:** [EDA Notebook](https://github.com/hrczggyrgy/ml-obesity-risk-prediction/blob/main/Notebooks/EDA_for_obesity_risk_notebook.ipynb)

## Model Training and Evaluation

Model training processes are thoroughly detailed in `scripts/train.py`. This script walks through the training of various models, evaluates their performance, and saves the best-performing models for later use. The training process and model evaluation are tracked and documented for transparency.

**Training Tracking:** [WandB Report](https://wandb.ai/herczeg-gyrgy/my-space/reports/Obesity-risk-prediction--Vmlldzo3MTg0MzI2)

**Model Evaluation Notebook:** [Final Model Notebook](https://github.com/hrczggyrgy/ml-obesity-risk-prediction/blob/main/Notebooks/final_model.ipynb)

## Web Application for Real-time Predictions

A Flask-based web application, set up in the `Streamlit/` directory, enables users to get real-time predictions. This interactive platform makes the project's findings accessible and practical for everyday use.

## Getting Started

To get the project up and running on your local machine, you'll need Python 3.x and the libraries listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt

'''bash

## Future Directions
The project lays a foundation for understanding the factors contributing to obesity. Moving forward, we plan to explore additional data processing techniques, more advanced feature engineering methods, and new model architectures to improve prediction accuracy and applicability.
